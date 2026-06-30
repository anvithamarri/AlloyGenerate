import os
import glob
import pandas as pd
import numpy as np
import pymatgen.core as mg
from scipy import stats
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry
from mp_api.client import MPRester
from chgnet.model.model import CHGNet
from chgnet.model import StructOptimizer

# ==========================================
# CONFIGURATION
# ==========================================
# FIXED — explicit, repo-relative paths
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))   # generation_model/validation/
REPO_ROOT    = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))  # AlloyGenerate/

CIF_FOLDER       = os.path.join(REPO_ROOT, "results", "act_gen_174")
TRAIN_CIF_FOLDER = os.path.join(REPO_ROOT, "cifs")
RESULTS_DIR      = os.path.join(REPO_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

df.to_csv(os.path.join(RESULTS_DIR, "alloy_generation_metrics_final.csv"), index=False)

filtered_df.to_csv(os.path.join(RESULTS_DIR, "filtered_alloy_metrics_final.csv"), index=False)

PHYSICAL_ENERGY_MIN = -7.0
PHYSICAL_ENERGY_MAX = -0.05
HULL_STABLE_CUTOFF  = 0.05
OUTLIER_HULL_CUTOFF = 5.0

print("Initializing CHGNet universal potential model...")
chgnet  = CHGNet.load()
relaxer = StructOptimizer(model=chgnet)
matcher = StructureMatcher(ltol=0.2, stol=0.2, angle_tol=5)
compat  = MaterialsProject2020Compatibility()

# ==========================================
# CIF REPAIR
# ==========================================
def load_and_repair_cif(file_path):
    fixed_lines = []
    with open(file_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) == 6 and (tokens[0] in ["Li", "Cd", "Os"] or tokens[2] == "1.0"):
                line = line.rstrip() + "  1.0000\n"
            fixed_lines.append(line)
    return mg.Structure.from_str("".join(fixed_lines), fmt="cif")

# ==========================================
# TRAINING SET (NOVELTY BASELINE)
# ==========================================
print("Loading training structures for novelty baseline...")
training_structures = []
if os.path.exists(TRAIN_CIF_FOLDER):
    for f in glob.glob(os.path.join(TRAIN_CIF_FOLDER, "*.cif")):
        try:
            training_structures.append(mg.Structure.from_file(f))
        except Exception:
            continue

if len(training_structures) == 0:
    raise RuntimeError(
        f"Training set folder '{TRAIN_CIF_FOLDER}' is empty or missing. "
        "Novelty metric cannot be computed without a baseline."
    )

train_formulas = set(s.composition.reduced_formula for s in training_structures)
print(f"Loaded {len(training_structures)} training structures ({len(train_formulas)} unique formulas)")

# ==========================================
# HULL FUNCTION
# ==========================================
def get_energy_above_hull(structure, ml_energy_per_atom, mpr):
    try:
        elements = [str(el) for el in structure.composition.elements]
        mp_entries = mpr.get_entries_in_chemsys(elements)
        if not mp_entries:
            return None
        mp_entries = compat.process_entries(mp_entries, on_error="ignore", verbose=False)
        if not mp_entries:
            return None
        total_ml_energy = ml_energy_per_atom * len(structure)
        user_entry = ComputedEntry(structure.composition, total_ml_energy)
        pd_diagram = PhaseDiagram(mp_entries + [user_entry])
        return pd_diagram.get_e_above_hull(user_entry)
    except Exception:
        return None

# ==========================================
# MAIN BATCH LOOP
# ==========================================
results_list = []
structure_cache = {}
cif_files = glob.glob(os.path.join(CIF_FOLDER, "*.cif"))

print(f"Processing {len(cif_files)} generated structures...")
mpr = MPRester(MP_API_KEY)

for index, cif_path in enumerate(cif_files):
    file_name = os.path.basename(cif_path)
    meta = {
        "File_Name": file_name, "Formula": "Invalid", "Is_Valid": False,
        "Is_Novel": True, "Raw_Energy_per_Atom": None, "Relaxed_Energy_per_Atom": None,
        "Energy_Drop": None, "Physically_Plausible": False, "E_Above_Hull": None,
    }

    # Pillar 1: Structural validity
    try:
        structure = load_and_repair_cif(cif_path)
        dm = structure.distance_matrix
        min_distance = min([dm[i][j] for i in range(len(dm)) for j in range(len(dm)) if i != j] or [99.0])
        if min_distance < 0.6:
            meta["Formula"] = f"{structure.composition.reduced_formula} (Clashed)"
            results_list.append(meta)
            continue
        meta["Is_Valid"] = True
        meta["Formula"] = structure.composition.reduced_formula
    except Exception:
        results_list.append(meta)
        continue

    # Pillar 2: Novelty vs training set
    for train_struct in training_structures:
        if matcher.fit(structure, train_struct):
            meta["Is_Novel"] = False
            break

    # Pillar 3: Energy + relaxation
    try:
        static_prediction = chgnet.predict_structure(structure)
        meta["Raw_Energy_per_Atom"] = float(static_prediction["e"])

        relaxation_result = relaxer.relax(structure, verbose=False)
        relaxed_structure = relaxation_result["final_structure"]
        final_total_energy = relaxation_result["trajectory"].energies[-1]
        relaxed_energy_per_atom = float(final_total_energy) / len(relaxed_structure)

        meta["Relaxed_Energy_per_Atom"] = relaxed_energy_per_atom
        meta["Energy_Drop"] = meta["Raw_Energy_per_Atom"] - relaxed_energy_per_atom

        is_plausible = PHYSICAL_ENERGY_MIN <= relaxed_energy_per_atom <= PHYSICAL_ENERGY_MAX
        meta["Physically_Plausible"] = is_plausible

        # Pillar 4: Hull distance (only for plausible structures)
        if is_plausible:
            structure_cache[file_name] = relaxed_structure
            ehull = get_energy_above_hull(relaxed_structure, relaxed_energy_per_atom, mpr)
            meta["E_Above_Hull"] = ehull

    except Exception as e:
        print(f"  CHGNet crashed on {file_name}: {e}")

    results_list.append(meta)
    if (index + 1) % 20 == 0 or (index + 1) == len(cif_files):
        print(f"  Processed {index + 1}/{len(cif_files)}")

mpr.session.close()

df = pd.DataFrame(results_list)
df["Novel_And_Plausible"] = df["Is_Novel"] & df["Physically_Plausible"]

# ==========================================
# INTERNAL DUPLICATE CHECK
# ==========================================
print("Checking internal duplicates...")
checked_structs = []
dup_flags = {}
for fname, struct in structure_cache.items():
    is_dup = any(matcher.fit(struct, other) for other in checked_structs)
    dup_flags[fname] = is_dup
    if not is_dup:
        checked_structs.append(struct)
df["Internal_Duplicate"] = df["File_Name"].map(dup_flags).fillna(False)

df.to_csv("alloy_generation_metrics_final.csv", index=False)

# ==========================================
# FINAL REPORT
# ==========================================
total = len(df)
valid = df["Is_Valid"].sum()
plausible = df["Physically_Plausible"].sum()
novel_plausible = df["Novel_And_Plausible"].sum()
unique_plausible = plausible - df["Internal_Duplicate"].sum()

plaus_energies = df.loc[df["Physically_Plausible"], "Relaxed_Energy_per_Atom"]
hull_values = df["E_Above_Hull"].dropna()
stable_count = (hull_values <= HULL_STABLE_CUTOFF).sum()
outliers = df[df["E_Above_Hull"] > OUTLIER_HULL_CUTOFF][["Formula", "E_Above_Hull"]].sort_values(
    "E_Above_Hull", ascending=False
)
hull_clean = hull_values[hull_values <= OUTLIER_HULL_CUTOFF]
trimmed_mean = stats.trim_mean(hull_values, 0.1) if not hull_values.empty else float("nan")

print("\n" + "=" * 55)
print("FINAL EVALUATION METRICS REPORT")
print("=" * 55)
print(f"Total Evaluated Structures:        {total}")
print(f"Valid (parseable after repair):    {valid} / {total} ({valid/total*100:.1f}%)")
print(f"Physically Plausible (of valid):   {plausible} / {valid} ({plausible/max(1,valid)*100:.1f}%)")
print(f"Novel & Plausible:                 {novel_plausible} / {plausible} ({novel_plausible/max(1,plausible)*100:.1f}%)")
print(f"Unique (no internal duplicates):   {unique_plausible} / {plausible}")
print("-" * 55)
print(f"Mean Relaxed Energy:    {plaus_energies.mean():.4f} eV/atom")
print(f"Median Relaxed Energy:  {plaus_energies.median():.4f} eV/atom")
print(f"Std Dev:                {plaus_energies.std():.4f} eV/atom")
print(f"5th Percentile:         {plaus_energies.quantile(0.05):.4f} eV/atom")
print(f"95th Percentile:        {plaus_energies.quantile(0.95):.4f} eV/atom")
print("-" * 55)
print(f"Hull-Evaluated Structures:         {len(hull_values)} / {plausible}")
print(f"Stable/Near-stable (<=0.05 eV/at): {stable_count} ({stable_count/max(1,len(hull_values))*100:.1f}%)")
print(f"Median E_above_hull:               {hull_values.median():.4f} eV/atom")
print(f"Mean E_above_hull:                 {hull_values.mean():.4f} eV/atom")
print(f"10% Trimmed Mean E_above_hull:     {trimmed_mean:.4f} eV/atom")
print(f"Mean E_above_hull (outliers >5 excluded, n={len(hull_clean)}): {hull_clean.mean():.4f} eV/atom")
print(f"Outliers (E_hull > {OUTLIER_HULL_CUTOFF} eV/atom): {len(outliers)}")
if not outliers.empty:
    print(outliers.to_string(index=False))
print("-" * 55)
excluded = valid - plausible
below = (df.loc[df["Is_Valid"], "Relaxed_Energy_per_Atom"] < PHYSICAL_ENERGY_MIN).sum()
above = (df.loc[df["Is_Valid"], "Relaxed_Energy_per_Atom"] > PHYSICAL_ENERGY_MAX).sum()
print(f"Excluded as unphysical: {excluded} ({below} below {PHYSICAL_ENERGY_MIN}, {above} above {PHYSICAL_ENERGY_MAX} eV/atom)")
print("=" * 55)
print(f"Training set: {len(training_structures)} structures, {len(train_formulas)} unique formulas")
print("Saved: alloy_generation_metrics_final.csv")
print("=" * 55)

# ==========================================
# TOP DISCOVERIES TABLE
# ==========================================
filtered_df = df[
    (df["Is_Valid"]) & (df["Physically_Plausible"]) & (~df["Internal_Duplicate"])
].copy()
filtered_df.to_csv("filtered_alloy_metrics_final.csv", index=False)

if not filtered_df.empty:
    unique_systems = filtered_df.loc[filtered_df.groupby("Formula")["Relaxed_Energy_per_Atom"].idxmin()]
    stable_table = unique_systems[unique_systems["E_Above_Hull"] <= HULL_STABLE_CUTOFF].sort_values("E_Above_Hull")
    print("\n" + "=" * 55)
    print(f"STABLE / NEAR-STABLE DISCOVERIES (n={len(stable_table)})")
    print("=" * 55)
    print(stable_table[["Formula", "Relaxed_Energy_per_Atom", "E_Above_Hull", "File_Name"]].to_string(index=False))
    print("=" * 55)