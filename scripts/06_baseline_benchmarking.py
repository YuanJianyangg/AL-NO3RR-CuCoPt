from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import warnings

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from tabpfn import TabPFNRegressor
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")


DATA_FILE = "initial_dataset_63.xlsx"

ID_COLS = ["Num.", "Materials"]
TARGET_COLS = ["Conversion", "Selectivity"]

OUTER_SPLITS = 5
INNER_SPLITS = 3
RANDOM_STATE = 42

TABPFN_DEVICE = "cuda"
TABPFN_N_ESTIMATORS = 64


class MultiOutputTabPFNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, device="cuda", n_estimators=64, random_state=42):
        self.device = device
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.x_scaler_ = StandardScaler()
        self.y_scaler_ = StandardScaler()

        X_scaled = self.x_scaler_.fit_transform(X)
        y_scaled = self.y_scaler_.fit_transform(y)

        self.models_ = []

        for j in range(y_scaled.shape[1]):
            model = TabPFNRegressor(
                device=self.device,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )
            model.fit(X_scaled, y_scaled[:, j])
            self.models_.append(model)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        X_scaled = self.x_scaler_.transform(X)

        predictions = [model.predict(X_scaled) for model in self.models_]
        predictions = np.column_stack(predictions)

        return self.y_scaler_.inverse_transform(predictions)


class MultiOutputFusedTabPFNGPRRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, device="cuda", n_estimators=64, random_state=42, gp_kernel=None):
        self.device = device
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.gp_kernel = gp_kernel or (
            C(1.0, (1e-3, 1e3))
            * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1e1))
        )

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.x_scaler_ = StandardScaler()
        self.y_scaler_ = StandardScaler()

        X_scaled = self.x_scaler_.fit_transform(X)
        y_scaled = self.y_scaler_.fit_transform(y)

        self.tabpfn_models_ = []

        with ThreadPoolExecutor() as executor:
            futures = []

            for j in range(y_scaled.shape[1]):
                model = TabPFNRegressor(
                    device=self.device,
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                )
                futures.append(
                    executor.submit(
                        self._fit_tabpfn_and_get_embedding,
                        model,
                        X_scaled,
                        y_scaled[:, j],
                    )
                )
                self.tabpfn_models_.append(model)

            train_embeddings = [future.result() for future in futures]

        fused_features = np.concatenate(train_embeddings, axis=1)

        base_gp = GaussianProcessRegressor(
            kernel=self.gp_kernel,
            alpha=1e-6,
            n_restarts_optimizer=10,
            random_state=self.random_state,
        )

        y_conversion = y_scaled[:, 0]
        y_selectivity = y_scaled[:, 1]

        self.selectivity_model_ = clone(base_gp)
        self.selectivity_model_.fit(fused_features, y_selectivity)

        conversion_X = np.hstack([fused_features, y_selectivity.reshape(-1, 1)])

        self.conversion_model_ = clone(base_gp)
        self.conversion_model_.fit(conversion_X, y_conversion)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        X_scaled = self.x_scaler_.transform(X)

        test_embeddings = []

        for model in self.tabpfn_models_:
            embeddings_3d = model.get_embeddings(X_scaled, data_source="test")
            embeddings_2d = np.mean(embeddings_3d, axis=0)
            test_embeddings.append(embeddings_2d)

        fused_features = np.concatenate(test_embeddings, axis=1)

        pred_selectivity = self.selectivity_model_.predict(fused_features)
        conversion_X = np.hstack([fused_features, pred_selectivity.reshape(-1, 1)])
        pred_conversion = self.conversion_model_.predict(conversion_X)

        pred_scaled = np.column_stack([pred_conversion, pred_selectivity])

        return self.y_scaler_.inverse_transform(pred_scaled)

    def _fit_tabpfn_and_get_embedding(self, model, X, y):
        model.fit(X, y)
        embeddings_3d = model.get_embeddings(X, data_source="test")
        embeddings_2d = np.mean(embeddings_3d, axis=0)
        return embeddings_2d


def build_models():
    gp_kernel = (
        C(1.0, (1e-3, 1e3))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-10, 1e1))
    )

    return {
        "TabPFN-GPR": MultiOutputFusedTabPFNGPRRegressor(
            device=TABPFN_DEVICE,
            n_estimators=TABPFN_N_ESTIMATORS,
            random_state=RANDOM_STATE,
            gp_kernel=gp_kernel,
        ),
        "TabPFN": MultiOutputTabPFNRegressor(
            device=TABPFN_DEVICE,
            n_estimators=TABPFN_N_ESTIMATORS,
            random_state=RANDOM_STATE,
        ),
        "GPR": MultiOutputRegressor(
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        GaussianProcessRegressor(
                            kernel=gp_kernel,
                            alpha=1e-6,
                            n_restarts_optimizer=10,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            )
        ),
        "Ridge": MultiOutputRegressor(
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", Ridge(random_state=RANDOM_STATE)),
                ]
            )
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor()),
            ]
        ),
        "SVR": MultiOutputRegressor(
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", SVR(kernel="rbf")),
                ]
            )
        ),
        "RandomForest": RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": MultiOutputRegressor(
            GradientBoostingRegressor(random_state=RANDOM_STATE)
        ),
        "XGBoost": MultiOutputRegressor(
            XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
    }


def param_grid(model_name):
    grids = {
        "Ridge": {
            "model__alpha": [0.1, 1.0, 10.0, 100.0],
        },
        "KNN": {
            "model__n_neighbors": [3, 5, 7, 9],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
        "SVR": {
            "model__C": [1.0, 5.0, 20.0, 100.0],
            "model__epsilon": [0.01, 0.05, 0.1],
            "model__gamma": ["scale", 0.03, 0.1],
        },
        "RandomForest": {
            "n_estimators": [200, 400],
            "max_depth": [None, 8, 16],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"],
        },
        "GradientBoosting": {
            "n_estimators": [120, 240],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [2, 3],
            "subsample": [0.8, 1.0],
        },
        "XGBoost": {
            "n_estimators": [200, 400],
            "max_depth": [3, 5],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "reg_lambda": [1.0, 5.0],
        },
    }

    return grids.get(model_name, {})


def make_strat_labels(y):
    return np.asarray(
        pd.cut(
            pd.Series(y).reset_index(drop=True),
            bins=5,
            labels=False,
            duplicates="drop",
            include_lowest=True,
        )
    )


def single_target_estimator(model):
    if isinstance(model, MultiOutputRegressor):
        return clone(model.estimator)

    return clone(model)


def tune_model(model_name, model, X, Y):
    grid = list(ParameterGrid(param_grid(model_name)))

    labels = make_strat_labels(Y["Conversion"])
    cv = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    best_params = {}

    for target_idx, target_name in enumerate(TARGET_COLS):
        y = Y.iloc[:, target_idx].to_numpy()

        best_score = -np.inf
        best_param = None

        for params in grid:
            fold_scores = []

            for train_idx, valid_idx in cv.split(X, labels):
                estimator = single_target_estimator(model)
                estimator.set_params(**params)

                estimator.fit(X.iloc[train_idx], y[train_idx])
                pred = np.asarray(estimator.predict(X.iloc[valid_idx])).reshape(-1)

                fold_scores.append(r2_score(y[valid_idx], pred))

            score = float(np.mean(fold_scores))

            if score > best_score:
                best_score = score
                best_param = params.copy()

        best_params[target_name] = best_param

    return best_params


def safe_corr(y_true, y_pred, method):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return np.nan

    if method == "pearson":
        return float(pearsonr(y_true, y_pred)[0])

    return float(spearmanr(y_true, y_pred)[0])


def calc_metrics(y_true, y_pred):
    return {
        "Spearman_rho": safe_corr(y_true, y_pred, "spearman"),
        "Pearson_r": safe_corr(y_true, y_pred, "pearson"),
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def print_rankings(metrics_df):
    for target in TARGET_COLS:
        print("\n" + "=" * 70)
        print(f"Target: {target}")
        print("=" * 70)

        target_df = metrics_df[metrics_df["target"] == target].copy()

        for metric in ["Spearman_rho", "Pearson_r", "R2", "RMSE"]:
            ascending = metric == "RMSE"
            ranked = target_df.sort_values(metric, ascending=ascending).reset_index(drop=True)
            ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

            print(f"\n[{metric} ranking]")
            print(
                ranked[["rank", "model", metric]]
                .to_string(index=False, float_format=lambda x: f"{x:.4f}")
            )


def main():
    repo_root = Path(__file__).resolve().parents[1]
    data_path = repo_root / "data" / DATA_FILE

    df = pd.read_excel(data_path, engine="openpyxl")

    feature_cols = [
        col for col in df.columns
        if col not in ID_COLS + TARGET_COLS
    ]

    X = df[feature_cols].copy()
    Y = df[TARGET_COLS].copy()

    models = build_models()

    labels = make_strat_labels(Y["Conversion"])
    outer_cv = StratifiedKFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_predictions = {
        model_name: np.full(Y.shape, np.nan, dtype=float)
        for model_name in models
    }

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, labels), start=1):
        print(f"\nFold {fold}/{OUTER_SPLITS}")

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        Y_train = Y.iloc[train_idx]

        for model_name, model in models.items():
            print(f"  {model_name}")

            grid = param_grid(model_name)

            if grid:
                best_params = tune_model(model_name, model, X_train, Y_train)
                pred_cols = []

                for target_idx, target_name in enumerate(TARGET_COLS):
                    estimator = single_target_estimator(model)
                    estimator.set_params(**best_params[target_name])

                    estimator.fit(X_train, Y_train.iloc[:, target_idx])
                    pred_target = np.asarray(estimator.predict(X_test)).reshape(-1)

                    pred_cols.append(pred_target)

                pred = np.column_stack(pred_cols)

            else:
                estimator = clone(model)
                estimator.fit(X_train, Y_train)
                pred = np.asarray(estimator.predict(X_test))

            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)

            oof_predictions[model_name][test_idx, :] = pred

    records = []

    for model_name, pred in oof_predictions.items():
        for target_idx, target_name in enumerate(TARGET_COLS):
            y_true = Y.iloc[:, target_idx].to_numpy()
            y_pred = pred[:, target_idx]

            records.append(
                {
                    "model": model_name,
                    "target": target_name,
                    **calc_metrics(y_true, y_pred),
                }
            )

    metrics_df = pd.DataFrame(records)
    print_rankings(metrics_df)


if __name__ == "__main__":
    main()