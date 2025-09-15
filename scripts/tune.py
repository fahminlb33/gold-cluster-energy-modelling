import argparse
import warnings
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
from matplotlib.figure import Figure

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

import optuna
import mlflow
from astartes import train_test_split


def remove_dict_key_prefix(d: dict[str, Any], prefix: str):
    return {k.replace(prefix, ""): v for k, v in d.items() if k.startswith(prefix)}


def get_or_create_experiment(experiment_name):
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)


class Objective:
    def __init__(
        self,
        experiment_id: int,
        dataset_path: str,
        model_name="stacking",
        split_method="random",
        random_seed=21,
        train_size=0.8,
        cv_n_splits=5,
        soap_features=10,
        pca_n_components=6,
        r_cut=5,
        n_max=1,
        l_max=7,
    ):
        self.experiment_id = experiment_id
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.split_method = split_method

        self.random_seed = random_seed
        self.train_size = train_size
        self.cv_n_splits = cv_n_splits
        self.soap_features = soap_features
        self.pca_n_components = pca_n_components
        self.r_cut = r_cut
        self.n_max = n_max
        self.l_max = l_max

        self.load_dataset()

    def __call__(self, trial: optuna.Trial):
        with mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=f"trial-{trial.number}",
            nested=True,
        ):
            params = self.sample_params(trial)
            metrics_val = {
                "val_r2": [],
                "val_mae": [],
                "val_rmse": [],
            }

            cv = KFold(
                n_splits=self.cv_n_splits, shuffle=True, random_state=self.random_seed
            )
            for i, (train_index, val_index) in enumerate(
                cv.split(self.X_train, self.y_train)
            ):
                X_train, y_train = self.X_train[train_index], self.y_train[train_index]
                X_val, y_val = self.X_train[val_index], self.y_train[val_index]

                model = self.get_model(params)
                model.fit(X_train, y_train)

                y_pred_val = model.predict(X_val)
                metrics_val["val_r2"].append(r2_score(y_val, y_pred_val))
                metrics_val["val_mae"].append(mean_absolute_error(y_val, y_pred_val))
                metrics_val["val_rmse"].append(
                    root_mean_squared_error(y_val, y_pred_val)
                )

                mlflow.log_figure(
                    self.plot_actual_predicted(y_val, y_pred_val),
                    f"val-{i + 1}-reg.png",
                )

            model = self.get_model(params)
            model.fit(self.X_train, self.y_train)
            y_pred_test = model.predict(self.X_test)

            mlflow.log_params(params)
            mlflow.log_metrics(
                {
                    "val_r2": np.mean(metrics_val["val_r2"]),
                    "val_mae": np.mean(metrics_val["val_mae"]),
                    "val_rmse": np.mean(metrics_val["val_rmse"]),
                    "test_r2": r2_score(self.y_test, y_pred_test),
                    "test_mae": mean_absolute_error(self.y_test, y_pred_test),
                    "test_rmse": root_mean_squared_error(self.y_test, y_pred_test),
                }
            )

            mlflow.log_figure(
                self.plot_actual_predicted(self.y_test, y_pred_test), "test-reg.png"
            )

        # optimize validation RMSE
        return np.mean(metrics_val["val_rmse"])

    def plot_actual_predicted(self, y_true, y_pred):
        fig = Figure()
        ax = fig.subplots()

        ax.scatter(y_true, y_pred)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")

        return fig

    def load_dataset(self):
        df = pd.read_csv(self.dataset_path)
        X = df.iloc[:, : self.soap_features].values
        y = df["energy"].values

        if self.split_method == "random" or self.split_method == "kennard_stone":
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X,
                y,
                sampler=self.split_method,
                train_size=self.train_size,
                random_state=self.random_seed,
            )
        elif self.split_method == "kmeans":
            self.X_train, self.X_test, self.y_train, self.y_test, _, _ = (
                train_test_split(
                    X,
                    y,
                    sampler=self.split_method,
                    train_size=self.train_size,
                    random_state=self.random_seed,
                )
            )
        else:
            raise ValueError("Invalid split method")

    def sample_params(self, trial: optuna.Trial) -> dict[str, Any]:
        if self.model_name == "extra_trees":
            return {
                "random_state": 21,
                "n_estimators": 100,
                "max_depth": trial.suggest_int("max_depth", 3, 10),  # tree depth
                "max_features": trial.suggest_float(
                    "max_features", 0.4, 1.0
                ),  # subsample
            }
        elif self.model_name == "xgboost":
            return {
                "random_state": 21,
                "booster": "gbtree",
                "tree_method": "exact",
                "n_estimators": 100,
                "max_depth": trial.suggest_int("max_depth", 3, 10),  # tree depth
                "eta": trial.suggest_float("eta", 1e-3, 1.0, log=True),  # learning rate
                "alpha": trial.suggest_float(
                    "alpha", 1e-3, 1.0, log=True
                ),  # L1 regularization
                "lambda": trial.suggest_float(
                    "lambda", 1e-3, 1.0, log=True
                ),  # L2 regularization
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.4, 1.0
                ),  # colsample_bytree
                "subsample": trial.suggest_float("subsample", 0.4, 1.0),  # subsample
            }
        elif self.model_name == "lightgbm":
            return {
                "verbose": -1,
                "random_state": 21,
                "boosting": "gbdt",
                "num_iterations": 100,
                "max_depth": trial.suggest_int("max_depth", 3, 10),  # tree depth
                "eta": trial.suggest_float("eta", 1e-3, 1.0, log=True),  # learning rate
                "lambda_l1": trial.suggest_float(
                    "lambda_l1", 1e-3, 10.0, log=True
                ),  # L1 regularization
                "lambda_l2": trial.suggest_float(
                    "lambda_l2", 1e-3, 10.0, log=True
                ),  # L2 regularization
                "feature_fraction": trial.suggest_float(
                    "feature_fraction", 0.4, 1.0
                ),  # colsample_bytree
                "bagging_fraction": trial.suggest_float(
                    "bagging_fraction", 0.4, 1.0
                ),  # subsample
            }
        elif self.model_name == "stacking":
            return {
                "random_state": 21,
                # extra-trees
                "etr_n_estimators": 100,
                "etr_max_depth": trial.suggest_int(
                    "etr_max_depth", 3, 10
                ),  # tree depth
                "etr_max_features": trial.suggest_float(
                    "etr_max_features", 0.4, 1.0
                ),  # subsample
                # xgboost
                "xgb_booster": "gbtree",
                "xgb_tree_method": "exact",
                "xgb_n_estimators": 100,
                "xgb_max_depth": trial.suggest_int(
                    "xgb_max_depth", 3, 10
                ),  # tree depth
                "xgb_eta": trial.suggest_float(
                    "xgb_eta", 1e-3, 1.0, log=True
                ),  # learning rate
                "xgb_alpha": trial.suggest_float(
                    "xgb_alpha", 1e-3, 1.0, log=True
                ),  # L1 regularization
                "xgb_lambda": trial.suggest_float(
                    "xgb_lambda", 1e-3, 1.0, log=True
                ),  # L2 regularization
                "xgb_colsample_bytree": trial.suggest_float(
                    "xgb_colsample_bytree", 0.4, 1.0
                ),  # colsample_bytree
                "xgb_subsample": trial.suggest_float(
                    "xgb_subsample", 0.4, 1.0
                ),  # subsample
                # lightgbm
                "lgb_verbose": -1,
                "lgb_boosting": "gbdt",
                "lgb_num_iterations": 100,
                "lgb_max_depth": trial.suggest_int(
                    "lgb_max_depth", 3, 10
                ),  # tree depth
                "lgb_eta": trial.suggest_float(
                    "lgb_eta", 1e-3, 1.0, log=True
                ),  # learning rate
                "lgb_lambda_l1": trial.suggest_float(
                    "lgb_lambda_l1", 1e-3, 10.0, log=True
                ),  # L1 regularization
                "lgb_lambda_l2": trial.suggest_float(
                    "lgb_lambda_l2", 1e-3, 10.0, log=True
                ),  # L2 regularization
                "lgb_feature_fraction": trial.suggest_float(
                    "lgb_feature_fraction", 0.4, 1.0
                ),  # colsample_bytree
                "lgb_bagging_fraction": trial.suggest_float(
                    "lgb_bagging_fraction", 0.4, 1.0
                ),  # subsample
            }

        raise ValueError("Unknown model name!")

    def get_model(self, params: dict[str, Any]) -> Pipeline:
        if self.model_name == "extra_trees":
            return Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    ("pca", PCA(n_components=self.pca_n_components)),
                    ("reg", ExtraTreesRegressor(**params)),
                ]
            )
        elif self.model_name == "xgboost":
            return Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    ("pca", PCA(n_components=self.pca_n_components)),
                    ("reg", XGBRegressor(**params)),
                ]
            )
        elif self.model_name == "lightgbm":
            return Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    ("pca", PCA(n_components=self.pca_n_components)),
                    ("reg", LGBMRegressor(**params)),
                ]
            )
        elif self.model_name == "stacking":
            return Pipeline(
                steps=[
                    ("scale", StandardScaler()),
                    ("pca", PCA(n_components=self.pca_n_components)),
                    (
                        "reg",
                        StackingRegressor(
                            estimators=[
                                (
                                    "etr",
                                    ExtraTreesRegressor(
                                        **remove_dict_key_prefix(params, "etr_")
                                    ),
                                ),
                                (
                                    "xgb",
                                    XGBRegressor(
                                        **remove_dict_key_prefix(params, "xgb_"),
                                    ),
                                ),
                                (
                                    "lgb",
                                    LGBMRegressor(
                                        **remove_dict_key_prefix(params, "lgb_"),
                                    ),
                                ),
                            ],
                        ),
                    ),
                ]
            )

        raise ValueError("Unknown model name!")


def main(args):
    if args.tracking_url:
        mlflow.set_tracking_uri(args.tracking_url)

    experiment_id = get_or_create_experiment(
        f"au20-{args.model_name}-{args.split_method}"
    )
    mlflow.set_experiment(experiment_id=experiment_id)

    objective = Objective(
        experiment_id,
        args.dataset_path,
        model_name=args.model_name,
        split_method=args.split_method,
        random_seed=args.random_seed,
        train_size=args.train_size,
        cv_n_splits=args.cv_n_splits,
        soap_features=args.soap_features,
        pca_n_components=args.pca_n_components,
        r_cut=args.r_cut,
        n_max=args.n_max,
        l_max=args.l_max,
    )

    study = optuna.create_study(
        study_name=f"{args.model_name}-{args.split_method}",
        storage="sqlite:///optuna.db",
        sampler=optuna.samplers.TPESampler(seed=args.random_seed),
        load_if_exists=True,
        directions=["minimize"],
    )

    study.optimize(objective, n_trials=args.n_trials)


if __name__ == "__main__":
    matplotlib.use("Agg")
    warnings.simplefilter("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=200)
    parser.add_argument("--tracking_url", type=str, default="http://10.20.20.101:8010")
    parser.add_argument("--dataset_path", type=str, default="../data/gold-features.csv")
    parser.add_argument(
        "--model_name",
        type=str,
        default="stacking",
        choices=["extra_trees", "xgboost", "lightgbm", "stacking"],
    )
    parser.add_argument(
        "--split_method",
        type=str,
        default="random",
        choices=["random", "kennard_stone", "kmeans"],
    )

    parser.add_argument("--random_seed", type=int, default=21)
    parser.add_argument("--train_size", type=int, default=0.8)
    parser.add_argument("--cv_n_splits", type=int, default=5)
    parser.add_argument("--soap_features", type=int, default=10)
    parser.add_argument("--pca_n_components", type=int, default=6)
    parser.add_argument("--r_cut", type=int, default=5)
    parser.add_argument("--n_max", type=int, default=1)
    parser.add_argument("--l_max", type=int, default=9)

    args = parser.parse_args()
    main(args)
