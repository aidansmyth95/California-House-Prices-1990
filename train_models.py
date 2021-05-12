from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.svm import SVR
import pandas as pd
import joblib
from sklearn.model_selection import GridSearchCV
from scipy import stats

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


def train_sklearn_models(data_file='train_data_prepared.npy', housing_labels_file='train_labels.npy',
                        X_test_file='X_test.npy', y_test_file='y_test.npy'):

    # load data
    housing_prepared = np.load(data_file)
    housing_labels = np.load(housing_labels_file)
    X_test = np.load(X_test_file)
    y_test = np.load(y_test_file)

    #*************************************************
    # Linear regression - underfits the data
    #*************************************************
    lin_reg = LinearRegression()
    print('\nFitting linear regression model...')
    lin_reg.fit(housing_prepared, housing_labels)

    # let's try the full preprocessing pipeline on a few training instances
    some_data_prepared = housing_prepared[:5]
    some_labels = housing_labels[:5]
    print("Predictions:", lin_reg.predict(some_data_prepared))
    print("Labels:", list(some_labels))

    # evaluate fit on training data - underfits
    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print('Training data RMSE: {}'.format(lin_rmse))
    lin_mae = mean_absolute_error(housing_labels, housing_predictions)
    print('Training data MAE: {}'.format(lin_mae))

    #*************************************************
    # Decision Tree regression - overfits the data
    #*************************************************
    tree_reg = DecisionTreeRegressor(random_state=42)
    print('\nFitting Decision Tree regression model...')
    tree_reg.fit(housing_prepared, housing_labels)
    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    print('Training data RMSE: {}'.format(tree_rmse))

    #*************************************************
    # k-fold validaton to demonstrate that above models are overfitting
    #*************************************************
    print('\nPerforming k-fold validation with linear regresion model for 10 folds...')
    lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    display_scores(lin_rmse_scores)

    print('\nPerforming k-fold validation with decision tree regresion model for 10 folds...')
    scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                            scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores = np.sqrt(-scores)
    # now it doesn't look so good!
    display_scores(tree_rmse_scores)

    #*************************************************
    # Random Forest regression - does pretty good.
    #*************************************************
    # idea: train several decision trees on random subsets of features, and then avg predictions
    # This is an example of ensemble learning.
    forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    print('\nFitting RandomForestRegressor model...')
    forest_reg.fit(housing_prepared, housing_labels)
    housing_predictions = forest_reg.predict(housing_prepared)
    forest_mse = mean_squared_error(housing_labels, housing_predictions)
    forest_rmse = np.sqrt(forest_mse)
    print('Training data RMSE: {}'.format(forest_rmse))

    # k-fold validation
    print('\nPerforming k-fold validation with RandomForestRegressor model for 10 folds...')
    forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                    scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores = np.sqrt(-forest_scores)
    display_scores(forest_rmse_scores)
    pd.Series(forest_rmse_scores).describe()

    # save model with pkl
    joblib.dump(forest_reg, "my_random_forest_model.pkl")
    #loaded_model = joblib.load("my_random_forest_model.pkl")

    # there are more models we should try, but will skip for now ....
    # model is still overfitting, we may want to add regularization or increase train data


    #*************************************************
    # Epsilon-Support Vector Regression
    #*************************************************
    print('\nFitting Epsilon-Support Vector Regression model...')
    svm_reg = SVR(kernel="linear")
    svm_reg.fit(housing_prepared, housing_labels)
    housing_predictions = svm_reg.predict(housing_prepared)
    svm_mse = mean_squared_error(housing_labels, housing_predictions)
    svm_rmse = np.sqrt(svm_mse)
    print('Training data RMSE: {}'.format(svm_rmse))


    #*************************************************
    # Hyperparam tuning using GridSearchCV
    #*************************************************
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
    ]
    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                            scoring='neg_mean_squared_error',
                            return_train_score=True)
    print('\nFitting Random forest regressor model with GridSearchCV hyperparam tuning...')
    grid_search.fit(housing_prepared, housing_labels)
    print('Best params: {}'.format(grid_search.best_params_))
    print('Best estimator: {}'.format(grid_search.best_estimator_))

    # show all experiments' results
    cvres = grid_search.cv_results_
    print('Result of all experiments performed during GridSearchCV:')
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    df_cvres = pd.DataFrame(grid_search.cv_results_)
    print(df_cvres)


    #*************************************************
    # Hyperparam tuning using GridSearchCV
    #*************************************************
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint

    param_distribs = {
            'n_estimators': randint(low=1, high=200),
            'max_features': randint(low=1, high=8),
        }
    print('\nFitting Random forest regressor model with RandomizedSearchCV hyperparam tuning...')
    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                    n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
    rnd_search.fit(housing_prepared, housing_labels)
    print('Best params: {}'.format(rnd_search.best_params_))
    print('Best estimator: {}'.format(rnd_search.best_estimator_))

    # show all experiments' results
    cvres = rnd_search.cv_results_
    print('Result of all experiments performed during RandomizedSearchCV:')
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    df_cvres = pd.DataFrame(rnd_search.cv_results_)
    print(df_cvres)


    #*************************************************
    # Inspecting best models
    #*************************************************
    print('\nInspecting best models...')
    
    # feature importance from random forest regressors
    grid_feature_importances = grid_search.best_estimator_.feature_importances_
    rnd_feature_importances = rnd_search.best_estimator_.feature_importances_
    print('GridSearchCV best estimator feature importance: {}'.format(grid_feature_importances))
    print('RandomizedSearchCV best estimator feature importance: {}'.format(rnd_feature_importances))
    #TODO: can also display these importancies next to attributes if pipeline was present to extract attribs from.
    # Unfortunately I did not import it here ...

    # test best model with test data - first time we used test data!
    print('\nTesting best model with test dataset...')
    final_model = grid_search.best_estimator_
    final_predictions = final_model.predict(X_test)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print('Test data RMSE: {}'.format(final_rmse))

    # compute 95% confidence interval - how precise point estimate is
    confidence = 0.95
    squared_errors = (final_predictions - y_test) ** 2
    conf_interval = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                            loc=squared_errors.mean(),
                            scale=stats.sem(squared_errors)))
    print('95 pct confidence interval using t-scores: {}'.format(conf_interval))

    # alternative but same:
    m = len(squared_errors)
    mean = squared_errors.mean()
    tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
    tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
    t_conf_interval = np.sqrt(mean - tmargin), np.sqrt(mean + tmargin)
    # assert t_conf_interval == t_conf_interval

    # alternatively we could use z-scores instead of t-scores
    zscore = stats.norm.ppf((1 + confidence) / 2)
    zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
    z_conf_interval = np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)
    print('95 pct confidence interval using z-scores: {}'.format(z_conf_interval))




if __name__ == '__main__':
    train_sklearn_models()
    print('Complete')