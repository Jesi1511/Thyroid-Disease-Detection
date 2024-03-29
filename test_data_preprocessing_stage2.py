import pandas as pd
import pytest
from sklearn.preprocessing import OrdinalEncoder
from step0_utility_functions import Utility
import os

def preprocess_data(data, encoder):
    # Assuming 'sex' is the column you want to encode
    data['sex'].fillna('Unknown', inplace=True)
    data['sex_encoded'] = encoder.transform(data[['sex']])
    # Add other preprocessing steps as needed
    return data

@pytest.fixture
def params():
    return Utility().read_params()

def test_check_saved_preprocess_pipelines(params):
    preprocess_pipe_foldername = params['preprocess']['preprocess_pipe_foldername']
    preprocess_pipe_filename = params['preprocess']['preprocess_pipe_filename']
    preprocess_pipe_path = os.path.join(preprocess_pipe_foldername, preprocess_pipe_filename)
    assert os.path.exists(preprocess_pipe_path)

def test_check_saved_label_encoder(params):
    preprocess_pipe_foldername = params['preprocess']['preprocess_pipe_foldername']
    le_transformer_filename = params['preprocess']['le_transformer_filename']
    le_file_path = os.path.join(preprocess_pipe_foldername, le_transformer_filename)
    assert os.path.exists(le_file_path)

def test_check_if_processed_data_is_saved(params):
    main_data_foldername = params['data_location']['main_data_folder']
    processed_stage2_data_foldername = params['data_location']['processed_stage2_data_foldername']
    processed_stage2_data_filename_X_train = params['data_location']['processed_stage2_data_filename_X_train']
    processed_stage2_data_filename_X_val = params['data_location']['processed_stage2_data_filename_X_val']
    processed_stage2_data_filename_y_train = params['data_location']['processed_stage2_data_filename_y_train']
    processed_stage2_data_filename_y_val = params['data_location']['processed_stage2_data_filename_y_val']
    assert os.path.exists(os.path.join(main_data_foldername, processed_stage2_data_foldername, processed_stage2_data_filename_X_train))
    assert os.path.exists(os.path.join(main_data_foldername, processed_stage2_data_foldername, processed_stage2_data_filename_X_val))
    assert os.path.exists(os.path.join(main_data_foldername, processed_stage2_data_foldername, processed_stage2_data_filename_y_train))
    assert os.path.exists(os.path.join(main_data_foldername, processed_stage2_data_foldername, processed_stage2_data_filename_y_val))

def test_check_if_interm_directories_are_removed(params):
    main_data_foldername = params['data_location']['main_data_folder']
    processed_stage1_data_foldername = params['data_location']['processed_stage1_data_foldername']
    interm1_data_foldername = params['data_location']['interm1_data_foldername']
    interm2_data_foldername = params['data_location']['interm2_data_foldername']
    assert not os.path.exists(os.path.join(main_data_foldername, interm1_data_foldername))
    assert not os.path.exists(os.path.join(main_data_foldername, interm2_data_foldername))
    assert not os.path.exists(os.path.join(main_data_foldername, processed_stage1_data_foldername))

def test_check_if_categorical_features_are_encoded(params):
    main_data_foldername = params['data_location']['main_data_folder']
    processed_stage2_data_foldername = params['data_location']['processed_stage2_data_foldername']
    processed_stage2_data_filename_X_train = params['data_location']['processed_stage2_data_filename_X_train']
    processed_stage2_data_filename_X_val = params['data_location']['processed_stage2_data_filename_X_val']
    processed_stage2_data_filename_y_train = params['data_location']['processed_stage2_data_filename_y_train']
    processed_stage2_data_filename_y_val = params['data_location']['processed_stage2_data_filename_y_val']

    # Initialize OrdinalEncoder with specified categories
    encoder = OrdinalEncoder(categories=[['Unknown', 'F', 'M']])

    X_train = pd.read_csv(os.path.join(main_data_foldername, processed_stage2_data_foldername, processed_stage2_data_filename_X_train), sep=',')
    X_val = pd.read_csv(os.path.join(main_data_foldername, processed_stage2_data_foldername, processed_stage2_data_filename_X_val), sep=',')
    y_train = pd.read_csv(os.path.join(main_data_foldername, processed_stage2_data_foldername, processed_stage2_data_filename_y_train), sep=',')
    y_val = pd.read_csv(os.path.join(main_data_foldername, processed_stage2_data_foldername, processed_stage2_data_filename_y_val), sep=',')

    # Apply preprocessing to the loaded data
    X_train = preprocess_data(X_train, encoder)
    X_val = preprocess_data(X_val, encoder)

    for feature in X_train:
        assert X_train[feature].dtypes != 'O'
        assert X_val[feature].dtypes != 'O'
    target_column_name = params['General']['target_column_name']
    assert y_train[target_column_name].dtypes != 'O'
    assert y_val[target_column_name].dtypes != 'O'

def test_check_if_the_missing_values_are_dealt_with(params):
    main_data_foldername = params['data_location']['main_data_folder']
    processed_stage2_data_foldername = params['data_location']['processed_stage2_data_foldername']
    processed_stage2_data_filename_X_train = params['data_location']['processed_stage2_data_filename_X_train']
    processed_stage2_data_filename_X_val = params['data_location']['processed_stage2_data_filename_X_val']
    X_train = pd.read_csv(os.path.join(main_data_foldername, processed_stage2_data_foldername, processed_stage2_data_filename_X_train), sep=',')
    X_val = pd.read_csv(os.path.join(main_data_foldername, processed_stage2_data_foldername, processed_stage2_data_filename_X_val), sep=',')
    for num in X_train.isnull().sum().values:
        if num > 0:
            assert False
    for num in X_val.isnull().sum().values:
        if num > 0:
            assert False
