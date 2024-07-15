from pyhocon import ConfigTree

from src.feature_engineering import FeatureEngineer
from src.feature_extraction import FeatureExtractor


class Search:
    def __init__(
        self, config: ConfigTree, feature_extractor: FeatureExtractor, feature_engineer: FeatureEngineer, model
    ) -> None:
        """
        Initialize a new instance of the Search class.

        Parameters:
        - config (ConfigTree): The configuration settings for the search.
        - feature_extractor (FeatureExtractor): An instance of the FeatureExtractor class for extracting features.
        - feature_engineer (FeatureEngineer): An instance of the FeatureEngineer class for engineering features.

        Returns:
        - None
        """
        self.config = config
        self.feature_extractor = feature_extractor
        self.feature_engineer = feature_engineer
        self.model = model

    def match(self, talent: dict, job: dict) -> dict:
        """
        This method takes a talent and job as input and uses the machine learning
        model to predict the label and a score.

        Args:
            talent: A dictionary representing a talent with relevant attributes.
            job: A dictionary representing a job with relevant attributes.

        Returns:
            A dictionary containing the talent, job, predicted label, and score.

        """
        extracted_features = self.feature_extractor.extract_features(talent, job)
        features = self.feature_engineer.engineer_features(extracted_features)
        score = self.model.predict_proba(features)[:, 1][0]
        label = 0 if score < 0.5 else 1
        return {"talent": talent, "job": job, "label": label, "score": score}

    def match_bulk(self, talents: list[dict], jobs: list[dict]) -> list[dict]:
        """
        This method takes a multiple talents and jobs as input and uses the machine
        learning model to predict the label for each combination.

        Args:
            talents: List of dicts representing talents with relevant attributes
            jobs: List of dicts representing jobs with relevant attributes

        Returns:
            list[dict]: A list of dictionaries, sorted by descending order of the score. Each dictionary
            contains the talent, job, predicted label, and score.
        """
        results = []
        for talent in talents:
            for job in jobs:
                result = self.match(talent, job)
                results.append(result)

        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        return sorted_results
