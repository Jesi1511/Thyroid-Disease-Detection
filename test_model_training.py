import json
import os
import pytest
import joblib
import pandas as pd
from step0_utility_functions import Utility
from sklearn.metrics import f1_score

@pytest.fixture
def params():
    """This fixture is used to load the params.yaml file."""
    return Utility().read_params()

class ModelUnderfitting(Exception):
    """Raised when the F1 score of the trained model is equal or less than 0.5."""
    pass

class ModelOverfitting(Exception):
    """Raised when the difference between the F1 score of train data and test data is more than or equal to 0.25."""
    pass

def test_check_the_saved_ml_model(params):
    """Check if the trained model is saved in the desired directory."""
    model_foldername = params['model']['model_foldername']
    model_name = params['model']['model_name']
    assert os.path.exists(os.path.join(model_foldername, model_name))

def test_check_metrics(params):
    """Check if the calculated metrics are saved in the desired directory and are valid."""
    metrics_folder = params['metrics_path']['metrics_folder']
    metrics_file = params['metrics_path']['metrics_file']

    with open(os.path.join(metrics_folder, metrics_file), 'r') as f:
        metrics = json.load(f)

    precision = metrics['precision_score']
    recall = metrics['recall_score']
    f1 = metrics['f1_score']

    assert os.path.exists(os.path.join(metrics_folder, metrics_file))
    assert 0 < precision < 1
    assert 0 < recall < 1
    assert 0 < f1 < 1

def test_check_for_underfitting_and_overfitting(params):
    """Check whether the trained model is overfitting or underfitting the data."""
    main_data_foldername = params['data_location']['main_data_folder']
    processed_stage2_data_foldername = params['data_location']['processed_stage2_data_foldername']
    processed_stage2_data_filename_X_train = params['data_location']['processed_stage2_data_filename_X_train']
    processed_stage2_data_filename_X_val = params['data_location']['processed_stage2_data_filename_X_val']
    processed_stage2_data_filename_y_train = params['data_location']['processed_stage2_data_filename_y_train']
    processed_stage2_data_filename_y_val = params['data_location']['processed_stage2_data_filename_y_val']

    X_train = pd.read_csv(os.path.join(main_data_foldername, processed_stage2_data_foldername, processed_stage2_data_filename_X_train))
    X_val = pd.read_csv(os.path.join(main_data_foldername, processed_stage2_data_foldername, processed_stage2_data_filename_X_val))
    y_train = pd.read_csv(os.path.join(main_data_foldername, processed_stage2_data_foldername, processed_stage2_data_filename_y_train))
    y_val = pd.read_csv(os.path.join(main_data_foldername, processed_stage2_data_foldername, processed_stage2_data_filename_y_val))

    model_foldername = params['model']['model_foldername']
    model_name = params['model']['model_name']
    
    model = joblib.load(os.path.join(model_foldername, model_name))

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    f1_train = f1_score(y_train, y_pred_train, average='weighted')
    f1_val = f1_score(y_val, y_pred_val, average='weighted')

    if f1_train <= 0.5:
        raise ModelUnderfitting

    if f1_train - f1_val >= 0.25:
        raise ModelOverfitting
