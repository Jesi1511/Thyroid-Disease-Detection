from step0_utility_functions import Utility
import pandas as pd
import os
import dill
import logging
import shutil

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Utility().create_folder('Logs')
params = Utility().read_params()

main_data_foldername = params['data_location']['main_data_folder']
processed_stage2_data_foldername = params['data_location']['processed_stage2_data_foldername']
processed_stage2_data_filename_X_train = params['data_location']['processed_stage2_data_filename_X_train']
processed_stage2_data_filename_X_val = params['data_location']['processed_stage2_data_filename_X_val']
processed_stage2_data_filename_y_train = params['data_location']['processed_stage2_data_filename_y_train']
processed_stage2_data_filename_y_val = params['data_location']['processed_stage2_data_filename_y_val']

class PreprocessStage3:

    def __init__(self) -> None:
        pass

    def preprocess3(self):
        try:
            # Load the processed data from Stage 2
            X_train = pd.read_csv(os.path.join(main_data_foldername, processed_stage2_data_foldername, processed_stage2_data_filename_X_train))
            X_val = pd.read_csv(os.path.join(main_data_foldername, processed_stage2_data_foldername, processed_stage2_data_filename_X_val))
            y_train = pd.read_csv(os.path.join(main_data_foldername, processed_stage2_data_foldername, processed_stage2_data_filename_y_train))
            y_val = pd.read_csv(os.path.join(main_data_foldername, processed_stage2_data_foldername, processed_stage2_data_filename_y_val))

            # Any additional preprocessing steps can be added here
            
            # Example: Impute missing values
            imputer = SimpleImputer(strategy='mean')
            X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
            X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)

            # Save the preprocessed data
            Utility().create_folder(os.path.join(main_data_foldername, 'preprocessed_data'))
            X_train.to_csv(os.path.join(main_data_foldername, 'preprocessed_data', 'X_train.csv'), index=False, sep=',')
            X_val.to_csv(os.path.join(main_data_foldername, 'preprocessed_data', 'X_val.csv'), index=False, sep=',')
            y_train.to_csv(os.path.join(main_data_foldername, 'preprocessed_data', 'y_train.csv'), index=False, sep=',')
            y_val.to_csv(os.path.join(main_data_foldername, 'preprocessed_data', 'y_val.csv'), index=False, sep=',')

            logger.info('Additional preprocessing completed and data saved.')

        except Exception as e:
            logger.error(e)
            raise e

if __name__ == "__main__":

    process3 = PreprocessStage3()
    process3.preprocess3()
    logger.info('Stage 4 Data preprocessing completed successfully.')
