"""
Module for training and serializing/deserialization of the machine learning model
"""

import logging

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None
import json

from joblib import dump, load
from pyhocon import ConfigTree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from src.feature_engineering import FeatureEngineer
from src.feature_extraction import FeatureExtractor

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self, config: ConfigTree, feature_extractor: FeatureExtractor, feature_engineer: FeatureEngineer
    ) -> None:
        """
        Initialize a new instance of the Train class.

        Parameters:
        - config (ConfigTree): The configuration settings

        Returns:
        - None
        """
        self.config = config
        self.feature_extractor = feature_extractor
        self.feature_engineer = feature_engineer
        self.model = None

    def training_pipeline(self, data_path: str, model_path: str) -> GradientBoostingClassifier:
        """
        This function is responsible for the complete training pipeline. It loads the data,
        creates training data, splits it into training and test sets, trains the model, evaluates it,
        and saves the trained model.

        Args:
            data_path: The path to the JSON file containing the training data.
            model_path: The path where the trained model will be saved.

        Returns:
            GradientBoostingClassifier: The trained model.
        """
        logger.info("Start data load")
        with open(data_path, "r") as file:
            data = json.load(file)
        logger.info("Finish data load")

        logger.info("Start creating training data")
        pdf_features, labels = self._create_training_data(data)
        logger.info(f"Finished creating training data: {len(pdf_features)} training examples produced ")
        X_train, X_test, y_train, y_test = self._train_test_split(pdf_features, labels)
        logger.info(f"Splitted data into {len(X_train)} examples and {len(X_test)} holdout test examples")
        logger.info("Start model training")
        self._train_model(X_train, y_train)
        logger.info("Finished model training")
        quality_metric = self._evaluate_model(X_test, y_test)
        logger.info(f"Accuracy measured on holdout test set: {quality_metric}")
        self._save_model(model_path)
        logger.info(f"Saved model to {model_path}")
        return self.model

    def _create_training_data(self, input_data: list[dict]) -> (pd.DataFrame, list):
        labels = []
        pdf_features = pd.DataFrame()
        for elem in input_data:
            labels.append(elem["label"])
            extracted_features = self.feature_extractor.extract_features(elem["talent"], elem["job"])
            features = self.feature_engineer.engineer_features(extracted_features)
            pdf_features = pd.concat([pdf_features, features], axis=0)
        return pdf_features, labels

    @staticmethod
    def _train_test_split(pdf_features: pd.DataFrame, labels: list) -> (pd.DataFrame, pd.DataFrame, list, list):
        X_train, X_test, y_train, y_test = train_test_split(
            pdf_features, labels, test_size=0.20, random_state=42, stratify=labels, shuffle=True
        )
        return X_train, X_test, y_train, y_test

    def _train_model(self, X_train: pd.DataFrame, y_train: list) -> None:
        self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42).fit(
            X_train, y_train
        )

    def _evaluate_model(self, X_test: pd.DataFrame, y_test: list) -> float:
        accuracy = self.model.score(X_test, y_test)
        return accuracy

    def _save_model(self, path):
        with open(path, "wb") as f:
            dump(self.model, f, protocol=5)

    def load_model(self, path):
        with open(path, "rb") as f:
            self.model = load(f)
            return self.model
