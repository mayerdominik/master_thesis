# Modeling Set Pieces in Soccer in an Expected Possession Value Framework – Bundesliga 2022/23 & 2023/24

This repository provides a machine learning pipeline to predict the success of set pieces (e.g., corners, free kicks) based on event data from the German Bundesliga.

Due to data protection agreements, raw event data is not included in this repository. Instead, the file `events_compressed_processed.pkl` is provided, which contains all precomputed feature values for set-piece events from the 2022/23 and 2023/24 Bundesliga seasons. It includes engineered feature representations, meta-information for each event, and the binary target variable `epv_success` (expected possession value success).

Model training, hyperparameter tuning, and feature importance interpretation are handled in the Jupyter notebook `model_training.ipynb`. The notebook loads the preprocessed data, trains multiple classification models (MLP, XGBoost, Random Forest), and analyzes the results using standard evaluation metrics and SHAP-based feature importance.

Required Python packages are listed in `requirements.txt`. The Python version used for development was **3.13.3**.

### Files Overview

- `events_compressed_processed.pkl`: Precomputed feature values and metadata for set-piece events.
- `model_training.ipynb`: Notebook for training, evaluation, and interpretation of models.
- `model_training.html`: HTML-Export of the jupyter notebook.
- `requirements.txt`: Contains all Python package dependencies.
- `video_goal_vs_augsburg.mp4`: Shows a successful corner kick possession.
