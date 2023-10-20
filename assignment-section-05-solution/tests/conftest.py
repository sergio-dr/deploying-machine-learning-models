import logging

import pytest
from sklearn.model_selection import train_test_split

from classification_model.config.core import config
from classification_model.processing.data_manager import load_raw_dataset

logger = logging.getLogger(__name__)


@pytest.fixture
def sample_input_data():
    data = load_raw_dataset(file_name=config.app_config.raw_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data,  
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        random_state=config.model_config.random_state,
    )

    return X_test
