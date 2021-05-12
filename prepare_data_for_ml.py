# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import numpy as np
# to make this notebook's output identical at every run
np.random.seed(42)

import os

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
import matplotlib.image as mpimg

import pandas as pd
from pandas.plotting import scatter_matrix


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    A custom transformer compatible with Scikit-Learn pipeline

    Adding TransformerMixin as base class gets fit_transform() method for free.
    BaseEstimator gets get_params() and set_params() methods for free, useful for hparam tuning.

    add_bedrooms_per_room is a hyperparam for the transformer. You can dd hparams for part of the pipeline
    that you are not sure about and would like to test.

    """

    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
        # column index
        #col_names = "total_rooms", "total_bedrooms", "population", "households"
        #rooms_ix, bedrooms_ix, population_ix, households_ix = [housing.columns.get_loc(c) for c in col_names]
        self.rooms_ix = 3
        self.bedrooms_ix = 4
        self.population_ix = 5
        self.households_ix = 6

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_per_household = X[:, self.rooms_ix] / X[:, self.households_ix]
        population_per_household = X[:, self.population_ix] / X[:, self.households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                        bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


if __name__ == '__main__':

    # read saved train data
    print('\nLoading train data...')
    strat_train_set = pd.read_pickle("train_data.pkl")
    print(strat_train_set.head(5))

    #***********************************************
    # Clean the dataset
    #***********************************************
    print('Extracting labels from DataFrame')
    housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    # check for null values
    sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
    print('Incomplete rows:')
    # it seems that it is "total_bedrooms" that can be incomplete in data. Easy to handle.
    print(sample_incomplete_rows)

    null_method = 1
    if null_method == 1:
        print('Removing samples that contain incomplete rows...')
        sample_incomplete_rows.dropna(subset=["total_bedrooms"])    # option 1
    elif null_method == 2:
        print('Removing total_bedrooms as a feature column for all samples...')
        sample_incomplete_rows.drop("total_bedrooms", axis=1)       # option 2
    elif null_method == 3:
        print('Using median value to replace null values...')
        median = housing["total_bedrooms"].median()
        sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # option 3
    else:
        raise Exception

    # another option (which we will use) is the Sklearn Imputer
    imputer = SimpleImputer(strategy="median")
    
    # we need to only keep numeric columns 
    housing_num = housing.select_dtypes(include=[np.number])
    # alternatively: housing_num = housing.drop("ocean_proximity", axis=1)
    
    print('Fitting Sklearn Imputer...')
    imputer.fit(housing_num)
    print('Imputer statistics: {}'.format(imputer.statistics_))
    print('Housing numeric median values used by imputation: {}'.format(housing_num.median().values))
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    print('Housing numeric samples that were previously missing values, post imputation:')
    print(housing_tr.loc[sample_incomplete_rows.index.values])
    print('Imputation startegy used: {}'.format(imputer.strategy))

    #***********************************************
    # Handling text and categorical attributes
    #***********************************************
    # only one categorical attribute - it is ordinal (proximity)
    housing_cat = housing[["ocean_proximity"]]
    print('\nHousing categorical attributes: {}'.format(housing_cat.head(10)))
    # fit ordinal encoder to the dataframe
    ordinal_encoder = OrdinalEncoder()
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    print('Incorrect ordinal encodings: {}'.format(housing_cat_encoded[:10]))
    print('Categories: {}'.format(ordinal_encoder.categories_))
    # we need to tell the ordinal encoder the correct ordering - at least I think this is correct order
    ordinal_encoder = OrdinalEncoder(categories=[np.array(['ISLAND', 'NEAR BAY', 'NEAR OCEAN', '<1H OCEAN', 'INLAND'])])
    housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
    print('Correct ordinal encodings: {}'.format(housing_cat_encoded[:10]))
    print('Categories: {}'.format(ordinal_encoder.categories_))

    # Maybe our data isn't ordinal - I mean some classes do seem to mean similar things in terms of proximity
    # Maybe we should stick to one hot encoding instead
    print('\nFitting one hot encoder instead...')
    cat_encoder = OneHotEncoder()
    housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
    # by default this is sparse array, but we can see full array using toarray()
    print(housing_cat_1hot.toarray())
    # If this list was very large, we would have too many encoded features.
    # If this happens, consider smarter attribute that is easier to encode, or using a low-dim vector called an embedding
    print('Categories: {}'.format(cat_encoder.categories_))


    #***********************************************
    # Build a custom transformer
    #***********************************************

    # customer transformer class
    print('Fitting custom transformer on data...')
    attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
    # this returns NumPy array
    housing_extra_attribs = attr_adder.transform(housing.values)
    # transform back to a dataframe
    housing_extra_attribs = pd.DataFrame(
        housing_extra_attribs,
        columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
        index=housing.index)
    print(housing_extra_attribs.head(5))


    #***********************************************
    # Build a pipeline to help with a sequence of transformations
    #***********************************************

    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")), # imputer
            ('attribs_adder', CombinedAttributesAdder()), # custom transformer
            # choice of std_scaler: unlike minmax it does not bound values,
            # but is less affected by outliers. Maybe less suitable for NNs.
            ('std_scaler', StandardScaler()), # feature scaling
        ])

    # apply this to numerical data only
    print('\nApplying numerical data pipeline...')
    housing_num_tr = num_pipeline.fit_transform(housing_num)
    print(housing_num_tr)

    # Scikit-Learn introduced CoulmnTransformer to handle both numerical and categorical together
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs), # resuse our numerical pipeline inside this one
        ("cat", OneHotEncoder(), cat_attribs), # just use one hot encoding
    ])
    # return numy array ready for ML
    print('\nFitting full final pipeline on data...')
    housing_prepared = full_pipeline.fit_transform(housing)
    print(housing_prepared)
    print(housing_prepared.shape)

    print('\nData is ML ready!')