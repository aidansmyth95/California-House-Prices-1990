import os
import tarfile
import urllib.request

def fetch_housing_data(housing_url, housing_path):
    """ A method to download California House Price dataset from 1990 census """

    # create housing path
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    
    # retrieve data from URL
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    
    # extract housing data from .tgz
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


if __name__ == '__main__':

    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    HOUSING_PATH = os.path.join("datasets", "housing")
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    print('\nFetching and extracting dataset...')
    fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)
    print('Dataset acquired.\n')

    # also download the California image
    images_path = os.path.join("images", "california_house_prices")
    os.makedirs(images_path, exist_ok=True)
    filename = "california.png"
    print("Downloading", filename)
    url = DOWNLOAD_ROOT + "images/end_to_end_project/" + filename
    urllib.request.urlretrieve(url, os.path.join(images_path, filename))