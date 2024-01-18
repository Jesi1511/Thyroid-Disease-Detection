from step0_utility_functions import Utility
import pandas as pd
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
LOG_FOLDER = 'Logs'
DATA_PREPROCESS_PATH = 'data_preprocess'
Utility().create_folder('Logs')
params = Utility().read_params()

main_data_foldername = params['data_location']['main_data_folder']
interm2_data_foldername = params['data_location']['interm2_data_foldername']
interm2_data_filename = params['data_location']['interm2_data_filename']
processed_stage1_data_foldername = params['data_location']['processed_stage1_data_foldername']
processed_stage1_data_filename = params['data_location']['processed_stage1_data_filename']

class PreprocessStage1:

    def __init__(self) -> None:
        self.main_data_foldername = params['data_location']['main_data_folder']

    def remove_unneccessary_columns(self, lst_of_useless_cols):
        try:
            df = pd.read_csv(os.path.join(self.main_data_foldername, 'raw_data', params['data_location']['raw_data_filename']))
            df_copy = df.copy()

            for feature in lst_of_useless_cols:
                df_copy.drop(feature, axis=1, inplace=True)

            interm1_data_foldername = params['data_location']['interm1_data_foldername']
            Utility().create_folder(os.path.join(self.main_data_foldername, interm1_data_foldername))

            df_copy.to_csv(os.path.join(self.main_data_foldername, interm1_data_foldername, params['data_location']['interm1_data_filename']), index=False, sep=',')
            logger.info('Removed unnecessary columns from the data.')

        except Exception as e:
            logger.error(e)
            raise e

    def converting_illogical_ages_to_null(self):
        try:
            main_data_foldername = params['data_location']['main_data_folder']
            interm1_data_foldername = params['data_location']['interm1_data_foldername']
            interm1_data_filename = params['data_location']['interm1_data_filename']
            interm2_data_foldername = params['data_location']['interm2_data_foldername']
            interm2_data_filename = params['data_location']['interm2_data_filename']

            # Reading the interm1 data
            df = pd.read_csv(os.path.join(main_data_foldername, interm1_data_foldername, interm1_data_filename))

            # Replacing the invalid ages with numpy nan values
            df['age'] = np.where(df['age'] > 100, np.nan, df['age'])
            df['age'] = np.where(df['age'] <= 0, np.nan, df['age'])

            # Creating a Data folder to save the processed data
            Utility().create_folder(main_data_foldername)
            Utility().create_folder(os.path.join(main_data_foldername, interm2_data_foldername))

            # Saving the processed data to the new folder
            df.to_csv(os.path.join(main_data_foldername, interm2_data_foldername, interm2_data_filename), index=False, sep=',')

            logger.info('Removed values of the column age that does not make sense.')

        except Exception as e:
            logger.error(e)
            raise e

    def replacing_dash_with_others_in_target_column(self):
        try:
            main_data_foldername = params['data_location']['main_data_folder']
            interm2_data_foldername = params['data_location']['interm2_data_foldername']
            interm2_data_filename = params['data_location']['interm2_data_filename']
            processed_stage1_data_foldername = params['data_location']['processed_stage1_data_foldername']
            processed_stage1_data_filename = params['data_location']['processed_stage1_data_filename']

            # Reading the interm2 data
            df = pd.read_csv(os.path.join(main_data_foldername, interm2_data_foldername, interm2_data_filename))

            # Replacing the dash ('-') from the target column with the 'Others' string
            df['target'] = np.where(df['target'] == '-', 'Others', df['target'])

            # Creating a Data folder to save the processed data
            Utility().create_folder(main_data_foldername)
            Utility().create_folder(os.path.join(main_data_foldername, processed_stage1_data_foldername))

            # Saving the processed data to the new folder
            df.to_csv(os.path.join(main_data_foldername, processed_stage1_data_foldername, processed_stage1_data_filename), index=False, sep=',')
            logger.info('Replaced dash with others in target column.')

        except Exception as e:
            logger.error(e)
            raise e

if __name__ == '__main__':

    lst = params['General']['list_of_columns_to_remove']

    process1 = PreprocessStage1()

    logger.info('Data preprocessing started.')
    process1.remove_unneccessary_columns(lst)
    process1.converting_illogical_ages_to_null()
    process1.replacing_dash_with_others_in_target_column()
    logger.info('Data preprocessing completed successfully.')
