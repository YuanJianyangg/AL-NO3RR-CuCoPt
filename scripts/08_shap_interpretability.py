from pathlib import Path
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from pygam import LinearGAM, s
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from tabpfn import TabPFNRegressor


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


DATA_FILE = f"iteration_6.xlsx"
MODEL_FILE = f"iteration_6_tabpfn_gpr_model.joblib"

TARGET_COLS = ["Conversion", "Selectivity"]

BACKGROUND_SIZE = 40
EXPLAIN_SAMPLE_SIZE = None
PERMUTATION_MAX_EVALS = 100
RANDOM_STATE = 42

TOP_K_PDP = 8
PDP_FEATURES = None

SHOW_BEESWARM = True
SHOW_IMPORTANCE = True
SHOW_PDP = True

GAM_N_SPLINES = None
GAM_SPLINE_RATIO = 0.60
GAM_MIN_SPLINES = 6
GAM_MAX_SPLINES = 18
GAM_SPLINE_ORDER = 3
GAM_LAM = 0.08
GAM_GRID_POINTS = 260
GAM_CI_WIDTH = 0.95

PDP_FIG_SIZE = (7.2, 4.8)
PDP_AX_FACE_COLOR = "#fcfdff"
PDP_POSITIVE_BG_COLOR = "#fdecef"
PDP_POSITIVE_BG_ALPHA = 0.55
PDP_NEGATIVE_BG_COLOR = "#eaf3ff"
PDP_NEGATIVE_BG_ALPHA = 0.55
PDP_POINT_SIZE = 28
PDP_POINT_COLOR = "#5a6f8f"
PDP_POINT_ALPHA = 0.42
PDP_FIT_LINE_COLOR = "#1d3557"
PDP_FIT_LINE_WIDTH = 2.6
PDP_CI_COLOR = "#4c78a8"
PDP_CI_ALPHA = 0.22
PDP_ZERO_LINE_COLOR = "#5f6c80"
PDP_ZERO_LINE_WIDTH = 1.25
PDP_ZERO_LINE_STYLE = (0, (5, 4))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 8,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})


class MultiOutputFusedTabPFN(RegressorMixin, BaseEstimator):
    def __init__(self, tabpfn_kwargs=None, gp_kernel=None, random_state=42):
        self.tabpfn_kwargs = tabpfn_kwargs
        self.gp_kernel = gp_kernel
        self.random_state = random_state
        self.tabpfn_models_ = []
        self.selectivity_model_ = None
        self.conversion_model_ = None

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        kwargs = dict(self.tabpfn_kwargs or {})
        self.tabpfn_models_ = []

        for target_idx in range(Y.shape[1]):
            model = TabPFNRegressor(**kwargs)
            model.fit(X, Y[:, target_idx])
            self.tabpfn_models_.append(model)

        train_embeddings = [
            self._get_embeddings(model, X)
            for model in self.tabpfn_models_
        ]
        fused_features = np.concatenate(train_embeddings, axis=1)

        kernel = self.gp_kernel
        if kernel is None:
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)

        self.selectivity_model_ = GaussianProcessRegressor(
            kernel=clone(kernel),
            normalize_y=True,
            random_state=self.random_state,
        )
        self.conversion_model_ = GaussianProcessRegressor(
            kernel=clone(kernel),
            normalize_y=True,
            random_state=self.random_state,
        )

        self.selectivity_model_.fit(fused_features, Y[:, 1])
        conversion_X = np.hstack([fused_features, Y[:, 1].reshape(-1, 1)])
        self.conversion_model_.fit(conversion_X, Y[:, 0])

        return self

    def predict(self, features, return_std=True):
        features = np.asarray(features, dtype=float)

        test_embeddings = [
            self._get_embeddings(model, features)
            for model in self.tabpfn_models_
        ]
        fused_features = np.concatenate(test_embeddings, axis=1)

        mean_sel, std_sel = self.selectivity_model_.predict(fused_features, return_std=True)

        conversion_X = np.hstack([fused_features, mean_sel.reshape(-1, 1)])
        mean_conv, std_conv = self.conversion_model_.predict(conversion_X, return_std=True)

        means = np.column_stack([mean_conv, mean_sel])
        stds = np.column_stack([std_conv, std_sel])

        if return_std:
            return means, stds
        return means

    def _get_embeddings(self, model, X):
        embeddings_3d = model.get_embeddings(X, data_source="test")
        embeddings_2d = np.mean(embeddings_3d, axis=0)
        return embeddings_2d


class TargetPredictor:
    def __init__(self, model, x_scaler, y_scaler, feature_cols, target_idx):
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.feature_cols = feature_cols
        self.target_idx = target_idx

    def __call__(self, x_raw):
        if isinstance(x_raw, pd.DataFrame):
            x_raw = x_raw[self.feature_cols].to_numpy(dtype=float)
        else:
            x_raw = np.asarray(x_raw, dtype=float)

        x_scaled = self.x_scaler.transform(x_raw)
        pred_scaled = self.model.predict(x_scaled, return_std=False)
        pred_raw = self.y_scaler.inverse_transform(pred_scaled)

        return pred_raw[:, self.target_idx]


def load_model_bundle(model_path):
    bundle = joblib.load(model_path)

    if isinstance(bundle, dict):
        model = bundle["model"]
        x_scaler = bundle["x_scaler"]
        y_scaler = bundle["y_scaler"]
        feature_cols = bundle["feature_cols"]
        target_cols = bundle.get("target_cols", TARGET_COLS)
        return model, x_scaler, y_scaler, feature_cols, target_cols

    model, x_scaler, y_scaler, feature_cols, target_cols = bundle
    return model, x_scaler, y_scaler, feature_cols, target_cols


def sample_rows(x_df, n):
    if n is None or n >= len(x_df):
        return x_df.copy()
    return x_df.sample(n=n, random_state=RANDOM_STATE)


def compute_shap_values(model, x_scaler, y_scaler, x_background, x_explain, feature_cols, target_idx):
    predictor = TargetPredictor(model, x_scaler, y_scaler, feature_cols, target_idx)
    explainer = shap.PermutationExplainer(predictor, x_background, seed=RANDOM_STATE)

    max_evals = max(PERMUTATION_MAX_EVALS, 2 * len(feature_cols) + 1)
    explanation = explainer(x_explain, max_evals=max_evals)

    values = np.asarray(explanation.values)
    if values.ndim == 3:
        values = values[:, :, 0]

    return values


def plot_beeswarm(shap_values, x_df, feature_cols, title):
    exp = shap.Explanation(
        values=shap_values,
        data=x_df[feature_cols].to_numpy(dtype=float),
        feature_names=feature_cols,
    )
    shap.plots.beeswarm(exp, max_display=20, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_importance(shap_values, feature_cols, title, top_n=20):
    importance = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(importance)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.barh(np.array(feature_cols)[order][::-1], importance[order][::-1], color="#4c78a8")
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title(title)
    ax.grid(axis="x", color="#d8dee9", linewidth=0.8, alpha=0.7)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def fit_gam_curve(x, y):
    unique_n = np.unique(x).size
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))

    if unique_n < 2 or x_min == x_max:
        span = max(0.5, abs(x_min) * 0.05)
        x_grid = np.array([x_min - span, x_max + span])
        y_trend = np.repeat(float(np.mean(y)), 2)
        y_low = np.repeat(float(np.percentile(y, 2.5)), 2)
        y_high = np.repeat(float(np.percentile(y, 97.5)), 2)
        return x_grid, y_trend, y_low, y_high

    if GAM_N_SPLINES is None:
        n_splines = int(round(unique_n * GAM_SPLINE_RATIO))
        n_splines = int(np.clip(n_splines, GAM_MIN_SPLINES, GAM_MAX_SPLINES))
    else:
        n_splines = int(GAM_N_SPLINES)

    n_splines = max(GAM_SPLINE_ORDER + 1, n_splines)

    x_grid = np.linspace(x_min, x_max, GAM_GRID_POINTS)
    gam = LinearGAM(
        s(0, n_splines=n_splines, spline_order=GAM_SPLINE_ORDER),
        lam=GAM_LAM,
    )
    gam.fit(x.reshape(-1, 1), y)

    y_trend = gam.predict(x_grid.reshape(-1, 1))
    ci = gam.confidence_intervals(x_grid.reshape(-1, 1), width=GAM_CI_WIDTH)

    return x_grid, y_trend, ci[:, 0], ci[:, 1]


def plot_shap_pdp_gam(x_df, shap_values, feature_cols, feature_name, target_name):
    idx = feature_cols.index(feature_name)
    x = x_df[feature_name].to_numpy(dtype=float)
    y = shap_values[:, idx]

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    x_grid, y_trend, y_low, y_high = fit_gam_curve(x, y)

    y_floor = float(np.nanmin([np.nanmin(y), np.nanmin(y_low), 0.0]))
    y_ceiling = float(np.nanmax([np.nanmax(y), np.nanmax(y_high), 0.0]))
    y_pad = 0.08 * max(1e-6, y_ceiling - y_floor)
    y_floor -= y_pad
    y_ceiling += y_pad

    fig, ax = plt.subplots(figsize=PDP_FIG_SIZE)
    ax.set_facecolor(PDP_AX_FACE_COLOR)

    ax.axhspan(0.0, y_ceiling, facecolor=PDP_POSITIVE_BG_COLOR, alpha=PDP_POSITIVE_BG_ALPHA, zorder=0)
    ax.axhspan(y_floor, 0.0, facecolor=PDP_NEGATIVE_BG_COLOR, alpha=PDP_NEGATIVE_BG_ALPHA, zorder=0)

    ax.scatter(
        x,
        y,
        s=PDP_POINT_SIZE,
        alpha=PDP_POINT_ALPHA,
        color=PDP_POINT_COLOR,
        edgecolors="none",
        zorder=3,
    )

    ax.fill_between(x_grid, y_low, y_high, color=PDP_CI_COLOR, alpha=PDP_CI_ALPHA, linewidth=0, zorder=2)
    ax.plot(x_grid, y_trend, color=PDP_FIT_LINE_COLOR, lw=PDP_FIT_LINE_WIDTH, zorder=4)

    ax.axhline(
        0.0,
        color=PDP_ZERO_LINE_COLOR,
        lw=PDP_ZERO_LINE_WIDTH,
        ls=PDP_ZERO_LINE_STYLE,
        zorder=5,
    )

    ax.set_ylim(y_floor, y_ceiling)
    ax.set_xlabel(feature_name)
    ax.set_ylabel("SHAP value")
    ax.set_title(f"SHAP dependence for {target_name.lower()}")

    for spine in ["left", "bottom", "top", "right"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#7b8794")
        ax.spines[spine].set_linewidth(1.0)

    ax.tick_params(axis="x", top=False)
    ax.tick_params(axis="y", right=False)
    ax.grid(axis="y", color="#d8dee9", alpha=0.6, linewidth=0.8)

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def select_pdp_features(shap_values, feature_cols):
    if PDP_FEATURES is not None:
        return [feature for feature in PDP_FEATURES if feature in feature_cols]

    importance = np.mean(np.abs(shap_values), axis=0)
    ranked_idx = np.argsort(importance)[::-1]
    return [feature_cols[i] for i in ranked_idx[:TOP_K_PDP]]


def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    df = pd.read_excel(DATA_DIR / DATA_FILE)
    model, x_scaler, y_scaler, feature_cols, target_cols = load_model_bundle(DATA_DIR / MODEL_FILE)

    x_all = df[feature_cols].astype(float)
    x_background = sample_rows(x_all, BACKGROUND_SIZE)
    x_explain = sample_rows(x_all, EXPLAIN_SAMPLE_SIZE)

    for target_idx, target_name in enumerate(target_cols):
        print(f"\nComputing SHAP for {target_name}...")

        shap_values = compute_shap_values(
            model=model,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            x_background=x_background,
            x_explain=x_explain,
            feature_cols=feature_cols,
            target_idx=target_idx,
        )

        if SHOW_BEESWARM:
            plot_beeswarm(
                shap_values=shap_values,
                x_df=x_explain,
                feature_cols=feature_cols,
                title=f"{target_name} SHAP beeswarm",
            )

        if SHOW_IMPORTANCE:
            plot_importance(
                shap_values=shap_values,
                feature_cols=feature_cols,
                title=f"{target_name} feature importance",
            )

        if SHOW_PDP:
            for feature_name in select_pdp_features(shap_values, feature_cols):
                plot_shap_pdp_gam(
                    x_df=x_explain,
                    shap_values=shap_values,
                    feature_cols=feature_cols,
                    feature_name=feature_name,
                    target_name=target_name,
                )


if __name__ == "__main__":
    main()
