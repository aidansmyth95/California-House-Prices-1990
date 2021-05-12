# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

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

# Where to save the figures
CHAPTER_ID = "california_house_prices"
IMAGES_PATH = os.path.join("images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    plt.clf()

def load_housing_data(housing_path):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


if __name__ == '__main__':

    # load dataframe
    HOUSING_PATH = os.path.join("datasets", "housing")
    housing = load_housing_data(housing_path=HOUSING_PATH)
    
    # visualize high-level dataframe info for overall data
    print(housing.head(5))
    print(housing.info())
    print(housing["ocean_proximity"].value_counts())
    print(housing.describe())

    # visualize histogram for each dataframe column value distribution for overall data
    housing.hist(bins=50, figsize=(20,15))
    save_fig("attribute_histogram_plots")

    # create test set
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    # OK, but someone just informed me that median income is a very important feature.
    # This would mean we want to do startified split to ensure that the test set is
    # representative of the entire dataset.
    # Since this column is continuous data, we need to forst make it categorical (binning it)
    print('Adding income_cat column...')
    housing["income_cat"] = pd.cut(housing["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf],
                                labels=[1, 2, 3, 4, 5])
    housing["income_cat"].hist(figsize=(20,15))
    save_fig("income_catattribute_histogram_plot")

    # startified sampling based on "income_cat" category
    print('\nStratified splitting by income_cat category...')
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    
    # lets check if it worked and how well it worked
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    compare_props = pd.DataFrame({
        "Overall": income_cat_proportions(housing),
        "Stratified": income_cat_proportions(strat_test_set),
        "Random": income_cat_proportions(test_set),
    }).sort_index()
    # create columns with Strat error as a %
    compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
    compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    print('\nStratified split stats:')
    print(compare_props)

    # With the dataset split appropriately, we can remove the income_cat column
    print('\nRemoving income_cat column...')
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    print(strat_train_set.head(5))

    #***********************************************
    # Explore data on map
    #***********************************************
    housing = strat_train_set.copy()
    # scatter plot longitude vs latitude to recreate a poor map of California
    housing.plot(kind="scatter", x="longitude", y="latitude")
    save_fig("bad_visualization_plot")

    # it is much easier to visualize scatter plots when high-density areas are clearer
    # Bay area, central valley, LA, SD are more visible now
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    save_fig("better_visualization_plot")

    # now lets look at house prices. Radius of circle represents districts population,
    # and the colour represents the price (jet ranges from blue to red)
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                s=housing["population"]/100, label="population", figsize=(16,10),
                c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
                sharex=False) #sharex to fix an axis bug
    plt.legend()
    # Seems that ocean proximity is a major factor (at least in SoCal)
    # Also seems that a clustering algorithm should do a fair job since high prices are often clustered
    save_fig("housing_prices_scatterplot")

    # an even better plot - Overlayed on California image
    california_img=mpimg.imread(os.path.join(IMAGES_PATH, 'california.png'))
    ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                    s=housing['population']/100, label="Population",
                    c="median_house_value", cmap=plt.get_cmap("jet"),
                    colorbar=False, alpha=0.4)
    plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
            cmap=plt.get_cmap("jet"))
    plt.ylabel("Latitude", fontsize=14)
    plt.xlabel("Longitude", fontsize=14)
    prices = housing["median_house_value"]
    tick_values = np.linspace(prices.min(), prices.max(), 11)
    cbar = plt.colorbar(ticks=tick_values/prices.max())
    cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
    cbar.set_label('Median House Value', fontsize=16)
    plt.legend(fontsize=16)
    save_fig("california_housing_prices_plot")


    #***********************************************
    # Explore data correlations
    #***********************************************
    corr_matrix = housing.corr()
    print('\nCorrelations in descending order for median_house_value:')
    # Downside - this only measures linear correlations, missing nonlinear relationships
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

    # pandas correlation plot for a select few attributes
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    # Looking at thes eplots, it is evident that some values are capped (median income, age, house value).
    # We may want to remove these districts where these quirks occur to avoid the pattern being reproduced by algorithms.
    save_fig("scatter_matrix_plot")

    # just look at median income vs median house value
    housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
    plt.axis([0, 16, 0, 550000])
    save_fig("income_vs_house_value_scatterplot")


    #***********************************************
    # Explore attribute combinations
    #***********************************************
    print('\Adding new attribute combinations...')
    housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
    housing["population_per_household"]=housing["population"]/housing["households"]
    # view updated correlation values
    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))

    housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value", alpha=0.2)
    plt.axis([0, 5, 0, 520000])
    save_fig("rooms_per_household_vs_house_value_scatterplot")
    print(housing.describe())

    # save the (cleaner) training and test dataframes
    strat_train_set.to_pickle("train_data.pkl")
    strat_test_set.to_pickle("test_data.pkl")
