# BCI
from BrainComputerInterface import BrainComputerInterface

# Dimensionality reduction algorithm
from CSP import CSP

# Classifiers and hyperparameter tuning
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    GridSearchCV,
    ShuffleSplit,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline

# Other
import numpy as np
import pandas as pd
import joblib
import os
import itertools
import warnings

warnings.filterwarnings("ignore")


N_COMPONENTS = 6


class BCITrainer(BrainComputerInterface):
    """
    Class for training machine learning models for BCI,
    using CSP and ensemble methods (bucket of models).

    Attributes
    ----------
    subject : int
        Subject number
    runs_motor : list[int]
        List of motor runs
    runs_imaginary : list[int]
        List of imaginary runs
    n_components : int
        Number of components to keep after CSP
    classifiers : tuple[VotingClassifier, dict]
        Tuple of classifiers and their hyperparameters
    pipeline : Pipeline
        Pipeline for training and evaluating the model

    Methods
    -------
    _init_ensemble()
        Initializes classifiers and their hyperparameters
    _hyperparameter_fit(classifier, param_grid, X, y)
        Fits hyperparameters for a classifier
    _tune_hyperparameters(classifiers, param_grid, X, y)
        Tunes hyperparameters for all classifiers
    _ensemble_method(voting_clf, classifiers, transformed_data, labels)
        Evaluates classifiers using cross-validation
    _create_pipeline(best_classifier_name, classifiers)
        Creates a pipeline with the best classifier
    _evaluate_model(pipeline, epochs_data, labels)
        Evaluates the model using cross-validation
    _save_model(model_name, model, best_interval, fft)
        Saves the model to a .pkl file
    _log_stats(subject, best_model, best_interval, best_score, fft)
        Logs the model stats to a .csv file
    _print_stats(path)
        Prints the model stats from a .csv file
    _find_best_interval_and_train_model()
        Finds the best interval and trains the model
    train()
        Trains the model
    """

    def __init__(self, subject: int, runs_motor: list[int], runs_imaginary: list[int]):
        super().__init__(subject, runs_motor, runs_imaginary)
        self.n_components = N_COMPONENTS

    @staticmethod
    def _init_ensemble() -> tuple[VotingClassifier, dict]:
        """
        Initializes classifiers and their hyperparameters
        Currently supported classifiers:
        - LogisticRegression
        - RandomForest
        - GradientBoosting
        - LDA
        - XGBoost

        Returns
        -------
        tuple[VotingClassifier, dict]
            Tuple of classifiers and their hyperparameters
        """
        param_grid = {
            "LogisticRegression": {"C": [0.01, 0.1, 1, 10, 100]},
            "RandomForest": {
                "n_estimators": [10, 50, 100, 200],
                "max_depth": [None, 10, 20, 30],
            },
            "GradientBoosting": {
                "learning_rate": [0.01, 0.1, 0.2],
                "n_estimators": [50, 100, 150],
            },
            "LDA": {"solver": ["svd", "lsqr", "eigen"]},
            "XGBoost": {"max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.2]},
        }

        classifiers = {
            "LogisticRegression": LogisticRegression(max_iter=10_000, solver="saga"),
            "RandomForest": RandomForestClassifier(),
            "GradientBoosting": GradientBoostingClassifier(),
            "LDA": LinearDiscriminantAnalysis(),
            "XGBoost": xgb.XGBClassifier(
                use_label_encoder=False, eval_metric="mlogloss"
            ),
        }

        return classifiers, param_grid

    @staticmethod
    def _hyperparameter_fit(classifier: Pipeline, param_grid: dict, X, y) -> Pipeline:
        """
        Fits hyperparameters for a classifier

        Parameters
        ----------
        classifier : Pipeline
            Classifier
        param_grid : dict
            Hyperparameters
        X : np.ndarray
            Data
        y : np.ndarray
            Labels

        Returns
        -------
        Pipeline
            Classifier with fitted hyperparameters
        """
        print(f"Tuning hyperparameters for {classifier.__class__.__name__}")
        cv = StratifiedKFold(n_splits=5)
        search = RandomizedSearchCV(classifier, param_grid, n_iter=10, cv=cv, n_jobs=-1)
        search.fit(X, y)
        return search.best_estimator_

    def _tune_hyperparameters(self, classifiers, param_grid, X, y):
        for clf_name, clf in classifiers.items():
            classifiers[clf_name] = self._hyperparameter_fit(
                clf, param_grid[clf_name], X, y
            )

        voting_clf = VotingClassifier(
            estimators=[(name, clf) for name, clf in classifiers.items()],
            voting="soft",
        )

        self.classifiers = (voting_clf, classifiers)

    @staticmethod
    def _ensemble_method(
        voting_clf: VotingClassifier,
        classifiers: dict,
        transformed_data: np.ndarray,
        labels: np.ndarray,
    ) -> str:
        """
        Evaluates classifiers using cross-validation

        Parameters
        ----------
        voting_clf : VotingClassifier
            Ensemble method classifier
        classifiers : dict
            Dictionary of classifiers
        transformed_data : np.ndarray
            Data in CSP space
        labels : np.ndarray
            Labels

        Returns
        -------
        str
            Name of the best classifier
        """
        print("-" * 80)
        print("Evaluating classifiers using cross-validation:")

        cv = ShuffleSplit(10, test_size=0.2, random_state=42)
        for clf_name, clf in classifiers.items():
            clf_scores = cross_val_score(
                clf, transformed_data, labels, cv=cv, n_jobs=-1
            )

            print(
                f"\nAccuracy for {clf_name}: %0.2f (+/- %0.2f)"
                % (clf_scores.mean(), clf_scores.std() * 2)
            )
            print(f"Classifier scores: {clf_scores}")

        ensemble_scores = cross_val_score(
            voting_clf, transformed_data, labels, cv=cv, n_jobs=-1
        )

        print("-" * 80)
        print(
            "Ensemble Method Accuracy: %0.2f (+/- %0.2f)"
            % (ensemble_scores.mean(), ensemble_scores.std() * 2)
        )
        print(f"Ensemble scores: {ensemble_scores}")

        best_classifier_name = max(
            classifiers,
            key=lambda clf: cross_val_score(
                classifiers[clf], transformed_data, labels, cv=cv, n_jobs=-1
            ).mean(),
        )

        print(f"Best classifier: {best_classifier_name}")

        return best_classifier_name

    def _create_pipeline(
        self, best_classifier_name: str, classifiers: dict
    ) -> Pipeline:
        """
        Creates a pipeline with the best classifier

        Parameters
        ----------
        best_classifier_name : str
            Name of the best classifier
        classifiers : dict

        Returns
        -------
        Pipeline
            Pipeline with the best classifier
        """
        best_classifier = classifiers[best_classifier_name]
        pipeline = Pipeline(
            [
                (
                    "csp",
                    CSP(n_components=self.n_components),
                ),
                ("clf", best_classifier),
            ]
        )
        return pipeline

    @staticmethod
    def _evaluate_model(
        pipeline: Pipeline, epochs_data: np.ndarray, labels: np.ndarray
    ) -> float:
        """
        Evaluates the model using cross-validation and prints the results

        Parameters
        ----------
        pipeline : Pipeline
            Pipeline with the best classifier
        epochs_data : np.ndarray
            Data
        labels : np.ndarray
            Labels

        Returns
        -------
        float
            Accuracy score
        """
        X_train, X_test, y_train, y_test = train_test_split(
            epochs_data, labels, stratify=labels, random_state=42
        )

        pipeline.fit(X_train, y_train)

        print("-" * 80 + "\n" + "-" * 80)
        print(f"Predictions for:")
        print(pipeline.predict(X_test))

        print(f"True labels:")
        print(y_test)

        score = pipeline.score(X_test, y_test)
        cvs = cross_val_score(pipeline, epochs_data, labels, cv=5, n_jobs=-1)

        print(
            f"Accuracy for {pipeline.named_steps['clf'].__class__.__name__} on validation data: {score}"
        )
        print(f"Cross validation scores: {cvs}")
        print("-" * 80)

        return score, pipeline

    @staticmethod
    def _save_model(
        model_name: str, model: Pipeline, best_interval: tuple[float], fft: bool
    ):
        """
        Saves the model to a .pkl file

        Parameters
        ----------
        model_name : str
            Name of the model
        model : Pipeline
            Pipeline with the best classifier
        best_interval : tuple[float]
            Best interval
        fft : bool
            Whether FFT was used

        Returns
        -------
        None
        """
        if not os.path.exists("models"):
            os.mkdir("models")
        print("Saving model as " + model_name)
        model_data = {"model": model, "best_interval": best_interval, "fft": fft}
        joblib.dump(model_data, f"models/{model_name}")

    def _log_stats(
        self,
        subject: int,
        best_model: str,
        best_interval: tuple[float],
        best_score: float,
        fft: bool,
    ):
        """
        Logs the model stats to a .csv file

        Parameters
        ----------
        subject : int
            Subject number
        best_model : str
            Name of the best classifier
        best_interval : tuple[float]
            Best interval
        best_score : float
            Best score
        fft : bool
            Whether FFT was used

        Returns
        -------
        None
        """
        if not os.path.exists("logs.csv"):
            with open("logs.csv", "w") as f:
                f.write("subject,task,best_model,best_interval,best_score,fft\n")
        with open("logs.csv", "a") as f:
            f.write(
                f"{subject},{self._get_tasks_number()},{best_model},{best_interval},{best_score},{fft}\n"
            )

    @staticmethod
    def _print_stats(path: str):
        """
        Prints the model stats from a .csv file

        Parameters
        ----------
        path : str
            Path to the .csv file

        Returns
        -------
        None
        """
        print(pd.read_csv(path).describe())

    def _find_best_interval_and_train_model(self):
        """
        Finds the best interval and trains the model.
        Saves the model to a .pkl file and logs the stats to a .csv file.

        Returns
        -------
        None
        """
        tmin_options = [-1.0, 0.0, 1.0]
        tmax_options = [2.0, 3.0, 4.0]
        fft_options = [True, False]
        csp_options = [2, 4, 6, 8, 16]

        print("*" * 80)
        print(f"Subject: {self.subject}")

        best_score = -1
        best_model = None
        best_interval = None
        best_fft = None

        for tmin, tmax, fft in itertools.product(
            tmin_options, tmax_options, fft_options
        ):
            if tmin >= tmax:
                continue
            self.raw_files = None
            self.classifiers = None
            self.pipeline = None

            print("-" * 80)
            print(f"Trying interval: {tmin} to {tmax}, fft: {fft}")
            epochs_data, labels = self._load_and_extract_features(
                self.subject, self.runs_motor, self.runs_imaginary, tmin, tmax, fft
            )
            transformed_data = CSP(n_components=self.n_components).fit_transform(
                epochs_data, labels
            )

            self._tune_hyperparameters(*self._init_ensemble(), transformed_data, labels)

            best_classifier_name = self._ensemble_method(
                self.classifiers[0], self.classifiers[1], transformed_data, labels
            )

            pipeline = self._create_pipeline(best_classifier_name, self.classifiers[1])

            score, model = self._evaluate_model(pipeline, epochs_data, labels)

            if score > best_score:
                best_score = score
                best_interval = (tmin, tmax)
                best_model = model
                best_fft = fft

        self.pipeline = best_model
        print(
            f"Best model: {best_model}, with score: {best_score}, on interval: {best_interval}, fft: {best_fft}"
        )

        self._save_model(
            f"subject_{self.subject}_task_{self._get_tasks_number()}.pkl",
            best_model,
            best_interval,
            best_fft,
        )

        self._log_stats(
            self.subject,
            self.pipeline.named_steps["clf"].__class__.__name__,
            best_interval,
            best_score,
            best_fft,
        )

        self._print_stats("logs.csv")

    def train(self):
        """
        Trains the model and prints the results, saves the model to a .pkl file

        Returns
        -------
        None
        """
        print("*" * 80)
        print(f"Train models for subject {self.subject}")
        print(f"Motor runs {self.runs_motor}")
        print(f"Imaginary runs {self.runs_imaginary}")
        self._find_best_interval_and_train_model()
