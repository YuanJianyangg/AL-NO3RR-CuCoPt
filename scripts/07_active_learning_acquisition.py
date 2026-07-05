from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler


MODEL_SOURCE = "initial_dataset_63"
# Manually change MODEL_SOURCE to:
# "initial_dataset_63", "iteration_1", "iteration_2", ..., "iteration_6"

PREDICT_FILE = "candidate_space_descriptors_13650.xlsx"
OUTPUT_FILE = f"13650_predictions_trained_on_{MODEL_SOURCE}.xlsx"

METAL_COLS = ["Cu", "Co", "Fe", "Ni", "Pd", "Ru", "Ag", "Mn", "Zn", "Pt", "Ir"]

N_SELECT = 10
XI = 1000
SIGMA = 5.0


def expected_improvement(pred_mean, pred_std, current_best, xi):
    pred_mean = np.asarray(pred_mean, dtype=float)
    pred_std = np.maximum(np.asarray(pred_std, dtype=float), 1e-12)

    improvement = pred_mean - current_best - xi
    z = improvement / pred_std

    return improvement * norm.cdf(z) + pred_std * norm.pdf(z)


def local_penalization(features, selected_indices, sigma):
    if len(selected_indices) == 0:
        return np.ones(len(features))

    selected_features = features[selected_indices]
    distances = cdist(features, selected_features, metric="euclidean")
    min_distances = np.min(distances, axis=1)
    penalty = np.exp(-(min_distances ** 2) / (2 * sigma ** 2))

    return 1 - penalty


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data"

    prediction_path = data_dir / OUTPUT_FILE
    experiment_path = data_dir / f"{MODEL_SOURCE}.xlsx"

    predict_data = pd.read_excel(prediction_path, engine="openpyxl")
    experiment_data = pd.read_excel(experiment_path, engine="openpyxl")

    predict_data = predict_data[
        ~predict_data["Materials"].isin(experiment_data["Materials"])
    ].reset_index(drop=True)

    predict_data["pred_Yield"] = (
        predict_data["pred_Conversion"] * predict_data["pred_Selectivity"]
    )

    predict_data["std_Yield"] = np.sqrt(
        (predict_data["pred_Selectivity"] * predict_data["std_Conversion"]) ** 2
        + (predict_data["pred_Conversion"] * predict_data["std_Selectivity"]) ** 2
        + (predict_data["std_Conversion"] * predict_data["std_Selectivity"]) ** 2
    )

    experiment_yield = experiment_data["Conversion"] * experiment_data["Selectivity"]
    current_best = experiment_yield.max()

    ei_values = expected_improvement(
        predict_data["pred_Yield"],
        predict_data["std_Yield"],
        current_best,
        xi=XI,
    )

    features = predict_data[METAL_COLS].to_numpy(dtype=float)
    features_scaled = StandardScaler().fit_transform(features)

    selected_indices = []
    lp_values = []
    penalized_ei_values = []

    for _ in range(N_SELECT):
        lp_factor = local_penalization(features_scaled, selected_indices, SIGMA)
        penalized_ei = ei_values * lp_factor

        if selected_indices:
            penalized_ei[selected_indices] = -np.inf

        best_idx = int(np.argmax(penalized_ei))

        selected_indices.append(best_idx)
        lp_values.append(lp_factor[best_idx])
        penalized_ei_values.append(penalized_ei[best_idx])

    result = predict_data.iloc[selected_indices].copy()
    result["EI"] = ei_values[selected_indices]
    result["LP_factor"] = lp_values
    result["Penalized_EI"] = penalized_ei_values

    output_cols = [
        "Materials",
        *METAL_COLS,
        "pred_Yield",
        "std_Yield",
        "EI",
        "LP_factor",
        "Penalized_EI",
    ]

    print(f"Prediction file: {OUTPUT_FILE}")
    print(f"Experimental data: {MODEL_SOURCE}.xlsx")
    print(f"Current best Yield: {current_best:.4f}")
    print(f"Remaining candidates: {len(predict_data)}")
    print(f"Selected samples: {N_SELECT}")
    print("=" * 120)
    print(result[output_cols].round(4).to_string(index=False))
    print("=" * 120)


if __name__ == "__main__":
    main()
