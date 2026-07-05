from pathlib import Path
from itertools import combinations

import pandas as pd


METALS = ("Co", "Fe", "Ni", "Pd", "Ru", "Ag", "Mn", "Zn", "Pt", "Ir")
COLUMNS = ("Cu", *METALS)

LOADING_STEP = 0.5
MAX_TOTAL_LOADING = 10.0


def loading_values(start, end):
    count = int((end - start) / LOADING_STEP) + 1
    return [start + i * LOADING_STEP for i in range(count)]


def empty_composition(cu):
    row = {"Cu": cu}

    for metal in METALS:
        row[metal] = 0.0

    return row


def generate_bimetal_space():
    rows = []

    for cu in loading_values(4.0, 9.5):
        for metal_loading in loading_values(0.5, 6.0):
            if cu + metal_loading <= MAX_TOTAL_LOADING:
                for metal in METALS:
                    row = empty_composition(cu)
                    row[metal] = metal_loading
                    rows.append(row)

    return rows


def generate_trimetal_space():
    rows = []

    for cu in loading_values(4.0, 9.0):
        for metal_1_loading in loading_values(0.5, 5.5):
            for metal_2_loading in loading_values(0.5, 5.5):
                if cu + metal_1_loading + metal_2_loading <= MAX_TOTAL_LOADING:
                    for metal_1, metal_2 in combinations(METALS, 2):
                        row = empty_composition(cu)
                        row[metal_1] = metal_1_loading
                        row[metal_2] = metal_2_loading
                        rows.append(row)

    return rows


def main():
    repo_root = Path(__file__).resolve().parents[1]
    output_path = repo_root / "data" / "candidate_space_13650.xlsx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = generate_bimetal_space() + generate_trimetal_space()
    candidate_space = pd.DataFrame(rows, columns=COLUMNS)
    candidate_space.to_excel(output_path, index=False)

    print(f"Saved {len(candidate_space)} candidates to {output_path}")


if __name__ == "__main__":
    main()