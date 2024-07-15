import logging

from test_data.test_data import jobs, talents

from src.create_config import create_config
from src.feature_engineering import FeatureEngineer
from src.feature_extraction import FeatureExtractor
from src.search import Search
from src.training import Trainer

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """
    Main function to perform matching of talents with jobs.
    """
    config = create_config("../../config")

    feature_extractor = FeatureExtractor(config)
    feature_engineer = FeatureEngineer(config)
    trainer = Trainer(config, feature_extractor, feature_engineer)
    model = trainer.load_model("../../model/model.joblib")

    search = Search(config, feature_extractor, feature_engineer, model)

    logger.info(f"Start matching of {len(talents)} talents with {len(jobs)} jobs")
    results = search.match_bulk(talents, jobs)
    logger.info(f"Matching results ordered in descending order by score: {results}")


if __name__ == "__main__":
    main()
