from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from tabpfn import TabPFNRegressor


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


MODEL_SOURCE = "initial_dataset_63"
# Manually change MODEL_SOURCE to:
# "initial_dataset_63", "iteration_1", "iteration_2", ..., "iteration_6"

PREDICT_FILE = "candidate_space_descriptors_13650.xlsx"
OUTPUT_FILE = f"13650_predictions_trained_on_{MODEL_SOURCE}.xlsx"

TARGET_COLS = ["Conversion", "Selectivity"]


class MultiOutputFusedTabPFN(RegressorMixin, BaseEstimator):
    def __init__(self, tabpfn_kwargs=None, gp_kernel=None, random_state=42):
        self.tabpfn_kwargs = tabpfn_kwargs or {}
        self.gp_kernel = gp_kernel or (
            C(1.0, (1e-3, 1e3))
            * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1e1))
        )
        self.random_state = random_state
        self.tabpfn_models_ = []
        self.selectivity_model_ = None
        self.conversion_model_ = None

    def fit(self, X, Y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not isinstance(Y, pd.DataFrame):
            Y = pd.DataFrame(Y, columns=TARGET_COLS)

        target_names = Y.columns
        self.tabpfn_models_ = []

        with ThreadPoolExecutor() as executor:
            futures = []

            for i, target_name in enumerate(target_names):
                y_i = Y.iloc[:, i]
                model = TabPFNRegressor(
                    **self.tabpfn_kwargs,
                    random_state=self.random_state,
                )
                futures.append(
                    executor.submit(
                        self._train_tabpfn_model,
                        X,
                        y_i,
                        target_name,
                        model,
                    )
                )
                self.tabpfn_models_.append(model)

            all_train_embeddings = [future.result() for future in futures]

        fused_features = np.concatenate(all_train_embeddings, axis=1)

        selectivity_y = Y.iloc[:, 1]
        conversion_y = Y.iloc[:, 0]

        base_gp_model = GaussianProcessRegressor(
            kernel=self.gp_kernel,
            alpha=1e-6,
            n_restarts_optimizer=10,
            random_state=self.random_state,
        )

        self.selectivity_model_ = clone(base_gp_model)
        self.selectivity_model_.fit(fused_features, selectivity_y)

        conversion_X = np.hstack(
            [fused_features, np.array(Y.iloc[:, 1]).reshape(-1, 1)]
        )

        self.conversion_model_ = clone(base_gp_model)
        self.conversion_model_.fit(conversion_X, conversion_y)

        return self

    def predict(self, features, return_std=True):
        test_embeddings = []

        for model in self.tabpfn_models_:
            embeddings_3d = model.get_embeddings(features, data_source="test")
            embeddings_2d = np.mean(embeddings_3d, axis=0)
            test_embeddings.append(embeddings_2d)

        fused_features = np.concatenate(test_embeddings, axis=1)

        n_samples = fused_features.shape[0]
        means = np.zeros((n_samples, 2))
        stds = np.zeros((n_samples, 2))

        mean, std = self.selectivity_model_.predict(
            fused_features,
            return_std=True,
        )
        means[:, 1] = mean
        stds[:, 1] = std

        X_gp_with_pred = np.hstack([fused_features, mean.reshape(-1, 1)])
        mean, std = self.conversion_model_.predict(
            X_gp_with_pred,
            return_std=True,
        )
        means[:, 0] = mean
        stds[:, 0] = std

        if return_std:
            return means, stds

        return means

    def _train_tabpfn_model(self, X, y_i, target_name, model):
        model.fit(X, y_i)
        embeddings_3d = model.get_embeddings(X, data_source="test")
        embeddings_2d = np.mean(embeddings_3d, axis=0)
        return embeddings_2d


def main():
    repo_root = Path(__file__).resolve().parents[1]

    model_path = repo_root / "data" / f"{MODEL_SOURCE}_tabpfn_gpr_model.joblib"
    predict_path = repo_root / "data" / PREDICT_FILE
    output_path = repo_root / "data" / OUTPUT_FILE

    saved = joblib.load(model_path)

    model = saved["model"]
    x_scaler = saved["x_scaler"]
    y_scaler = saved["y_scaler"]
    feature_cols = saved["feature_cols"]

    df = pd.read_excel(predict_path, engine="openpyxl")

    X_new = df[feature_cols].copy()
    X_new_scaled = x_scaler.transform(X_new)

    means_scaled, stds_scaled = model.predict(X_new_scaled, return_std=True)

    means = y_scaler.inverse_transform(means_scaled)
    stds = stds_scaled * y_scaler.scale_

    result = df.copy()
    result["pred_Conversion"] = means[:, 0]
    result["std_Conversion"] = stds[:, 0]
    result["pred_Selectivity"] = means[:, 1]
    result["std_Selectivity"] = stds[:, 1]

    result.to_excel(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()