import dask.dataframe as dd
import pytest
from step0_utility_functions import Utility

@pytest.fixture
def params():
    return Utility().read_params()

def test_check_if_data_is_loaded_as_dataframe(params):

    ## getting data url from params.yaml file
    initial_url = params['data_location']['data_url_base']
    data_url = params['data_location']['data_url']

    url = initial_url + data_url.split('/')[-2]

    ## Reading the csv file with dask for memory-efficient loading
    data = dd.read_csv(url)

    assert isinstance(data, dd.DataFrame)

