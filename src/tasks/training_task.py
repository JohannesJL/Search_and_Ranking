import logging

from src.create_config import create_config
from src.feature_engineering import FeatureEngineer
from src.feature_extraction import FeatureExtractor
from src.training import Trainer

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    """
    Main function to run the training pipeline.
    """
    config = create_config("../../config")
    feature_extractor = FeatureExtractor(config)
    feature_engineer = FeatureEngineer(config)
    trainer = Trainer(config, feature_extractor, feature_engineer)
    logger.info("Started training pipeline")
    trainer.training_pipeline(data_path="../../data/data.json", model_path="../../model/model.joblib")
    logger.info("Finished training pipeline")


if __name__ == "__main__":
    main()
