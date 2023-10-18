'''
Test the functions contained in:
  - src/data.py
  - src/model.py
'''

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from os.path import dirname
from os.path import abspath
import src.model as md
import src.data as dt
import pytest
import numpy as np
import pandas as pd


current_dir = dirname(abspath(__file__))


@pytest.fixture(scope='module')
def df():
    df = pd.read_csv(f'{current_dir}/data/census.csv')
    df = dt.clean_data(df)
    return df


@pytest.fixture(scope='module')
def cat_features(df):
    cat_features = df.select_dtypes(include=object).columns.tolist()
    cat_features.remove('salary')
    return cat_features


@pytest.fixture(scope='module')
def split_data(df):
    train, test = train_test_split(df, test_size=0.20, random_state=42)
    return train, test


@pytest.fixture(scope='module')
def train(split_data):
    return split_data[0]


@pytest.fixture(scope='module')
def test(split_data):
    return split_data[1]


def test_transform_data(train, test, cat_features, label='salary'):
    '''
    Tests the transform_data() function within src/data.py

    Inputs
    ------
    train : pd.DataFrame
        Training dataset.

    test : pd.DataFrame
        Validation dataset.

    cat_features : list[str]
        A list containing the categorical features.

    label : str
        The column to be used as label.

    '''
    try:
        X_train, y_train, encoder, lb = dt.process_data(
            train,
            categorical_features=cat_features,
            label="salary",
            training=True
        )
        X_test, y_test, encoder, lb = dt.process_data(
            test,
            categorical_features=cat_features,
            label="salary",
            encoder=encoder,
            lb=lb,
            training=False
        )
    except Exception:
        print("transform_data failed: function failed to run properly")

    try:
        assert X_train.shape[0] != 0
        assert y_train.shape[0] != 0
        assert X_test.shape[0] != 0
        assert y_test.shape[0] != 0
    except AssertionError as err:
        print("transform_data failed: no data is returned")
        raise err

    try:
        arr = np.array([])
        assert type(X_train) == type(arr)
        assert type(y_train) == type(arr)
        assert type(X_test) == type(arr)
        assert type(y_test) == type(arr)
    except AssertionError as err:
        print("transform_data failed: variables are not numpy arrays")
        raise err

    # assigning variables as pytest global variables
    pytest.X_train = X_train
    pytest.y_train = y_train
    pytest.X_test = X_test
    pytest.y_test = y_test


@pytest.fixture(scope='module')
def X_train():
    return pytest.X_train


@pytest.fixture(scope='module')
def y_train():
    return pytest.y_train


@pytest.fixture(scope='module')
def X_test():
    return pytest.X_test


@pytest.fixture(scope='module')
def y_test():
    return pytest.y_test


def test_train_model(X_train, y_train):
    '''
    Tests the train_model() function within src/model.py

    Inputs
    ------
    X_train : np.array
        Training data.

    y_train : np.array
        Training labels.
    '''
    try:
        model = md.train_model(X_train, y_train)
    except Exception:
        print('train_model failed: function failed to run properly')

    try:
        assert model is not None
    except AssertionError as err:
        print('train_model failed: no model is returned')
        raise err

    try:
        knn = KNeighborsClassifier()
        assert type(model) == type(knn)
    except AssertionError as err:
        print('train_model failed: model type is wrong')
        raise err

    pytest.model = model


def test_inference(X_test):
    '''
    Tests the inference() function within src/model.py

    Inputs
    ------
    X : np.array
        Data used for prediction.
    '''
    model = pytest.model

    try:
        preds = md.inference(model, X_test)
        assert preds.shape[0] != 0
    except AssertionError as err:
        print('inference failed: no predictions returned')
        raise err
