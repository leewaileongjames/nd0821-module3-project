'''
Test the functions contained in:
  - src/data.py
  - src/model.py
'''

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import src.model as md
import src.data as dt
import pytest
import numpy as np
import pandas as pd


@pytest.fixture(scope='module')
def df():
    df = pd.read_csv('data/census.csv')
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
    Test transform_data

    :param df: Clean data
    :param label: Name of label column
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
    except:
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
    Test train_model

    :param X_train: Training data
    :param y_train: Training labels
    '''
    try:
        model = md.train_model(X_train, y_train)
    except:
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
    Test inference function

    :param X_test: Test data
    '''
    model = pytest.model

    try:
        preds = md.inference(model, X_test)
        assert preds.shape[0] != 0
    except AssertionError as err:
        print('inference failed: no predictions returned')
        raise err

"""
@pytest.fixture(scope='module')
def feature():
    return 'education'


@pytest.fixture(scope='module')
def test(df):
    test = train_test_split(df, test_size=0.20, random_state=42)[1]
    return test


def test_slice_metrics(feature, test, y_test):
    '''
    Test slice_metrics function

    :param feature: feature to slice on
    :param df: features and label
    :param y_test: ground truth colums
    '''
    preds = pytest.preds
    try:
        metrics = ml.slice_metrics(feature, test, y_test, preds)
        assert metrics.empty is False
    except AssertionError as err:
        print('slice_metrics failed: no metrics generated')
        raise err

"""
