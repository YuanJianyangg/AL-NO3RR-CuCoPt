from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


METALS = ("Co", "Fe", "Ni", "Pd", "Ru", "Ag", "Mn", "Zn", "Pt", "Ir")
COLUMNS = ("Cu", *METALS)
N_SAMPLES = 60


def format_number(value):
    value = float(value)
    return str(int(value)) if value.is_integer() else str(value)


def format_formula(sample):
    formula = f"Cu{format_number(sample[0])}"

    for metal, value in zip(METALS, sample[1:]):
        if value > 0:
            formula += f"{metal}{format_number(value)}"

    return formula


def main():
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "data" / "candidate_space_13650.xlsx"

    candidate_space = pd.read_excel(input_path)
    all_combinations = candidate_space.loc[:, COLUMNS].to_numpy(dtype=float)

    dist_matrix = squareform(pdist(all_combinations, metric="euclidean"))

    selected_indices = [0]

    first_sample = 0
    dist_to_first = dist_matrix[first_sample]
    second_sample = np.argmax(dist_to_first)
    selected_indices.append(second_sample)

    while len(selected_indices) < N_SAMPLES:
        min_dists = np.min(dist_matrix[selected_indices], axis=0)
        next_sample = np.argmax(min_dists)
        selected_indices.append(next_sample)

    selected_samples = [all_combinations[i] for i in selected_indices]

    for sample in selected_samples:
        formula = format_formula(sample)
        vector = [float(value) for value in sample]
        print(f"{formula} {vector}")


if __name__ == "__main__":
    main()