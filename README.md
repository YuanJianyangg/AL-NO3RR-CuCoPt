# Code Availability: Prior-informed active learning discovers a Cu–Co–Pt reactive center for nitrate-to-ammonia electroreduction

This repository contains the data files and Python scripts used for composition-space generation, descriptor calculation, TabPFN-GPR modelling, baseline benchmarking, active-learning acquisition, and SHAP-based model interpretation for "Prior-informed active learning discovers a Cu–Co–Pt reactive center for nitrate-to-ammonia electroreduction".

## Data

The `data/` folder contains the input datasets and generated tables used by the scripts.

| File | Description |
| --- | --- |
| `candidate_space_13650.xlsx` | Candidate composition space containing 13,650 Cu-based bimetallic and trimetallic compositions. Columns are the 11 metal loadings. |
| `candidate_space_descriptors_13650.xlsx` | Candidate composition space with material names, 11 metal-loading features, and 21 composition-weighted physicochemical descriptors. |
| `initial_dataset_63.xlsx` | Initial experimental dataset with 63 samples. |
| `iteration_1.xlsx` to `iteration_6.xlsx` | Cumulative active-learning datasets for successive iterations. `iteration_6.xlsx` corresponds to the final experimental dataset included here. |


## Scripts

The scripts should be run sequentially from the repository root. The recommended order is:

Trained TabPFN-GPR model files are generated locally by `04_tabpfn_gpr_train.py` . Run the training script before prediction, acquisition, or SHAP interpretation scripts that load a trained model.

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
