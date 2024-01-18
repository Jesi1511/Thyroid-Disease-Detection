import pandas as pd
import pytest
from step0_utility_functions import Utility
from pathlib import Path
from sklearn.preprocessing import OrdinalEncoder
import sys
sys.path.append("path/to/Thyroid_Disease_Detection_Internship-main")
@pytest.fixture
def params():
    return Utility().read_params()

def preprocess_data(data, encoder):
    # Assuming 'sex' is the column you want to encode
    data['sex'].fillna('Unknown', inplace=True)
    data['sex_encoded'] = encoder.transform(data[['sex']])
    # Add other preprocessing steps as needed
    return data

def test_check_input_shape(params):

    """This python test is used to check the shape of input raw data."""

    main_data_folder = Path(params['data_location']['main_data_folder'])
    raw_data_folder = Path(params['data_location']['raw_data_folder'])
    raw_data_filename = Path(params['data_location']['raw_data_filename'])
    input_data = pd.read_csv(main_data_folder / raw_data_folder / raw_data_filename)

    # Initialize OrdinalEncoder with specified categories
    encoder = OrdinalEncoder(categories=[['Unknown', 'F', 'M']])
    input_data = preprocess_data(input_data, encoder)

    assert input_data.shape == (9172, 32)  # Assuming 'sex_encoded' is added

def test_check_output_shape(params):

    """This python test is used to check the shape of data after preprocessing performed."""

    main_data_folder = Path(params['data_location']['main_data_folder'])
    processed_stage2_data_foldername = Path(params['data_location']['processed_stage2_data_foldername'])
    processed_stage2_data_filename_X_train = Path(params['data_location']['processed_stage2_data_filename_X_train'])
    processed_stage2_data_filename_X_val = Path(params['data_location']['processed_stage2_data_filename_X_val'])
    processed_stage2_data_filename_y_train = Path(params['data_location']['processed_stage2_data_filename_y_train'])
    processed_stage2_data_filename_y_val = Path(params['data_location']['processed_stage2_data_filename_y_val'])

    X_train = pd.read_csv(main_data_folder / processed_stage2_data_foldername / processed_stage2_data_filename_X_train)
    X_val = pd.read_csv(main_data_folder / processed_stage2_data_foldername / processed_stage2_data_filename_X_val)
    y_train = pd.read_csv(main_data_folder / processed_stage2_data_foldername / processed_stage2_data_filename_y_train)
    y_val = pd.read_csv(main_data_folder / processed_stage2_data_foldername / processed_stage2_data_filename_y_val)

    # Initialize OrdinalEncoder with specified categories
    encoder = OrdinalEncoder(categories=[['Unknown', 'F', 'M']])
    X_train = preprocess_data(X_train, encoder)
    X_val = preprocess_data(X_val, encoder)
  

    assert X_train.shape == (8254, 24)  # Assuming 'sex_encoded' is added
    assert X_val.shape == (918, 24)  # Assuming 'sex_encoded' is added
    assert y_train.shape == (8254, 1)
    assert y_val.shape == (918, 1)
   