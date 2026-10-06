# Code Availability: Single-Atom Pt Couples Reactive-Hydrogen Generation with NOx Hydrogenation in Active-Learning-Discovered Cu–Co–Pt Ensembles

This repository contains the data files and Python scripts used for composition-space generation, descriptor calculation, TabPFN-GPR modelling, baseline benchmarking, active-learning acquisition, and SHAP-based model interpretation for "Single-Atom Pt Couples Reactive-Hydrogen Generation with NOx Hydrogenation in Active-Learning-Discovered Cu–Co–Pt Ensembles".

The TabPFN-GPR model concatenates embeddings from two target-specific TabPFN regressors. Both GPR regressors use these fused embeddings to predict `Conversion` and `Selectivity`, respectively, and provide predictive standard deviations.

## Data

The `data/` folder contains the input datasets and generated tables used by the scripts.

| File | Description |
| --- | --- |
| `candidate_space_13650.xlsx` | Candidate composition space containing 13,650 Cu-based bimetallic and trimetallic compositions. Columns are the 11 metal loadings. |
| `candidate_space_descriptors_13650.xlsx` | Candidate composition space with material names, 11 metal-loading features, and 21 composition-weighted physicochemical descriptors. |
| `initial_dataset_63.xlsx` | Initial experimental dataset with 63 samples. |
| `iteration_1.xlsx` to `iteration_6.xlsx` | Cumulative active-learning datasets for successive iterations. `iteration_6.xlsx` corresponds to the final experimental dataset included here. |
| `13650_predictions_trained_on_initial_dataset_63.xlsx` | Predictions and predictive uncertainties for all 13,650 candidates using the model trained on the initial dataset. |

## Scripts

Run the scripts from the repository root. Scripts `01`–`03` prepare the candidate data, which are already provided in `data/`. Scripts `04`–`05` train the model and generate candidate predictions. Benchmarking, acquisition, and SHAP interpretation can then be run as needed, subject to the inputs described below.

Trained TabPFN-GPR model files are generated locally by `04_tabpfn_gpr_train.py` and saved as `data/<dataset_stem>_tabpfn_gpr_model.joblib`. Prediction and SHAP interpretation load a trained model; acquisition reads the prediction table and matching experimental dataset.

| Script | Purpose |
| --- | --- |
| `01_generate_composition_space.py` | Generates the 13,650 Cu-based bimetallic and trimetallic candidate compositions and saves `candidate_space_13650.xlsx`. |
| `02_kennard_stone_sampling.py` | Performs Kennard-Stone sampling on the candidate composition space and prints the selected initial candidates. |
| `03_descriptor_generation.py` | Generates material names and 21 composition-weighted physicochemical descriptors using element properties from `mendeleev`. |
| `04_tabpfn_gpr_train.py` | Trains the TabPFN-GPR model on a selected dataset and writes the trained model locally. |
| `05_tabpfn_gpr_predict.py` | Loads a selected trained TabPFN-GPR model and predicts `Conversion` and `Selectivity` for the 13,650 candidates. |
| `06_baseline_benchmarking.py` | Benchmarks TabPFN-GPR against baseline regressors using out-of-fold metrics. |
| `07_active_learning_acquisition.py` | Computes predicted yield, uncertainty-propagated yield uncertainty, expected improvement, and local-penalization scores for candidate acquisition. |
| `08_shap_interpretability.py` | Computes SHAP values and shows SHAP beeswarm, feature-importance, and SHAP dependence plots. The default setting uses `iteration_6`, corresponding to the final full dataset. |

## Running the model and predictions

By default, training and prediction use `initial_dataset_63.xlsx`. To regenerate the model and the predictions for all 13,650 candidates, run:

```bash
python scripts/04_tabpfn_gpr_train.py
python scripts/05_tabpfn_gpr_predict.py
```

The scripts use CUDA and 64 TabPFN estimators. Training saves `data/initial_dataset_63_tabpfn_gpr_model.joblib`; prediction writes `data/13650_predictions_trained_on_initial_dataset_63.xlsx`. Rerunning these scripts replaces the corresponding generated files.
