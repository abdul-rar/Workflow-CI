import shutil
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load preprocessed data
TRAIN_PATH = 'energy_consumption_preprocessing/train_preprocessing.csv'
TEST_PATH = 'energy_consumption_preprocessing/test_preprocessing.csv'
TARGET = 'Energy Consumption'

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]

X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

# Define grid search params
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None]
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Start MLflow run (Wajib agar bisa log dan ambil run_id)
with mlflow.start_run():
    # Log params
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("best_n_estimators", best_model.n_estimators)
    mlflow.log_param("best_max_depth", best_model.max_depth)

    # Metrics - train
    y_train_pred = best_model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_train_pred)

    # Metrics - test
    y_test_pred = best_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)

    # Log metrics
    mlflow.log_metric("training_mean_absolute_error", train_mae)
    mlflow.log_metric("training_mean_squared_error", train_mse)
    mlflow.log_metric("training_root_mean_squared_error", train_rmse)
    mlflow.log_metric("training_r2_score", train_r2)
    mlflow.log_metric("training_score", train_r2)

    mlflow.log_metric("testing_mean_absolute_error", test_mae)
    mlflow.log_metric("testing_mean_squared_error", test_mse)
    mlflow.log_metric("testing_root_mean_squared_error", test_rmse)
    mlflow.log_metric("RandomForestRegressor_score_X_test", test_r2)

    # Save model
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        input_example=X_train.iloc[:5]
    )

    # # Copy artifacts to folder outside
    # EXPERIMENT_NAME = "Energy Consumption - Tuning"

    # run_id = mlflow.active_run().info.run_id
    # mlflow.set_experiment(EXPERIMENT_NAME)
    # experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    # experiment_id = experiment.experiment_id

    # model_source_path = os.path.join("mlruns", experiment_id, run_id, "artifacts", "model")
    # model_target_path = os.path.join("..", "artifacts")

    # os.makedirs(model_target_path, exist_ok=True)
    # for filename in os.listdir(model_source_path):
    #     shutil.copy(
    #         os.path.join(model_source_path, filename),
    #         os.path.join(model_target_path, filename)
    #     )
