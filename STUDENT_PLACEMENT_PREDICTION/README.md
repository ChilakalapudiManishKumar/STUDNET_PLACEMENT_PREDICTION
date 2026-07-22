# Student Placement Prediction — Run Order

Put these files in ONE folder together:
- pipeline.py
- eda.py
- prep_model_data.py
- train_models.py
- student_placement_dataset.csv   (your raw dataset)

Then run them in VS Code / terminal in this exact order:

```bash
python pipeline.py          # cleans raw data -> student_placement_cleaned.csv
python eda.py                # generates plots/ folder with 9 charts
python prep_model_data.py    # feature engineering, scaling, train/test split -> scaler.joblib, X_train.csv, etc.
python train_models.py       # trains 5-6 models, saves plots/ + model_comparison.csv + best_model.joblib
```

Each script checks for its required input file and gives a clear error telling you
which previous script to run first if something's missing.

XGBoost is optional — train_models.py will skip it automatically if not installed
(install with `pip install xgboost` to include it).
