import os
import pandas as pd
from pymatgen.core import Composition
from mp_api.client import MPRester
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility

MP_API_KEY = os.getenv("MP_API_KEY")
HULL_STABLE_CUTOFF = 0.05

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "generation_model", "results")

# ==========================================
# LOAD STABLE CANDIDATES FROM run_metrics.py OUTPUT
# ==========================================
filtered_df = pd.read_csv(os.path.join(RESULTS_DIR, "filtered_alloy_metrics_final.csv"))
stable_df = filtered_df[filtered_df["E_Above_Hull"] <= HULL_STABLE_CUTOFF].copy()
stable_formulas = stable_df["Formula"].tolist()

print(f"Verifying {len(stable_formulas)} stable/near-stable candidates against live Materials Project database...")

# ==========================================
# ROBUST NOVELTY CHECK — reduced composition matching, not formula string matching
# (string matching can miss equivalent compositions written differently)
# ==========================================
def check_novelty_robust(formula, mpr):
    comp = Composition(formula)
    elements = [str(el) for el in comp.elements]
    try:
        entries = mpr.get_entries_in_chemsys(elements)
    except Exception as e:
        return None, None, str(e)

    exact_matches = [
        e for e in entries
        if e.composition.reduced_composition == comp.reduced_composition
    ]
    return exact_matches, len(entries), None

mpr = MPRester(MP_API_KEY)
novelty_results = []

for formula in stable_formulas:
    matches, total_chemsys_entries, error = check_novelty_robust(formula, mpr)

    if error:
        print(f"{formula:<15} | check failed: {error}")
        novelty_results.append({
            "Formula": formula,
            "MP_Entries_Exact_Composition": None,
            "MP_Entries_In_Chemsys": None,
            "Lowest_MP_Energy_per_Atom": None,
            "Is_Novel_vs_MP": None
        })
        continue

    n_exact = len(matches)
    lowest_energy = min([e.energy_per_atom for e in matches]) if matches else None
    is_novel = n_exact == 0

    novelty_results.append({
        "Formula": formula,
        "MP_Entries_Exact_Composition": n_exact,
        "MP_Entries_In_Chemsys": total_chemsys_entries,
        "Lowest_MP_Energy_per_Atom": lowest_energy,
        "Is_Novel_vs_MP": is_novel
    })

    print(f"{formula:<15} | exact matches: {n_exact} | chemsys entries: {total_chemsys_entries} | Novel: {is_novel}")

mpr.session.close()

# ==========================================
# SAVE + SUMMARY
# ==========================================
novelty_df = pd.DataFrame(novelty_results)
novelty_df.to_csv(os.path.join(RESULTS_DIR, "mp_novelty_check.csv"), index=False)

n_novel = novelty_df["Is_Novel_vs_MP"].sum()
n_total = len(novelty_df)

print("\n" + "=" * 55)
print("MATERIALS PROJECT NOVELTY VERIFICATION")
print("=" * 55)
print(f"Total stable/near-stable candidates checked: {n_total}")
print(f"Genuinely novel (absent from MP):             {n_novel} / {n_total} ({n_novel/n_total*100:.1f}%)")
print(f"Rediscoveries (existing MP entries):           {n_total - n_novel} / {n_total}")

rediscoveries = novelty_df[novelty_df["Is_Novel_vs_MP"] == False]
if not rediscoveries.empty:
    print("\nRediscoveries — compare against your CHGNet-relaxed energy for validation:")
    print(rediscoveries[["Formula", "MP_Entries_Exact_Composition", "Lowest_MP_Energy_per_Atom"]].to_string(index=False))
print("=" * 55)
