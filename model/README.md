# Crevasse training data for [NZ Space Challenge](https://www.nzspacechallenge.com/)

This folder contains the Keras/Tensorflow training datasets for the github repository found [here](https://github.com/weiji14/nz_space_challenge).
This training dataset will be used to perform a supervised classification algorithm to detect crevasse lines in Antarctica from satellite imagery.

The dataset is prepared from this [data preparation](https://github.com/weiji14/nz_space_challenge/blob/master/data_prep.ipynb) jupyter notebook, and it will feed into the [crevasse finder](https://github.com/weiji14/nz_space_challenge/blob/master/crevasse_finder.ipynb) Convolutional Neural Network model.

Click on the [binder](https://mybinder.org) link button below to launch an interactive notebook!
The data will be loaded via [Quilt](https://github.com/quiltdata/quilt), using instructions from [data2binder](https://github.com/quiltdata/data2binder).

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/weiji14/nz_space_challenge/master)


## Description of dataset

### Training data (for crevasse classifier)

- X_data.npy (image) - Cropped image from Sentinel 2 data bands 4,3,2, True Color image retrieved from Sentinel Hub using gain=0.3 and gamma=0.8.
- Y_data.npy (mask) - Manually digitized crevasse polygons using leaflet.draw webtool [here](https://weiji14.github.io/nz_space_challenge).

![Sentinel2 image and crevasse lines](https://user-images.githubusercontent.com/23487320/39954410-0f461808-5613-11e8-977e-1a1fd742d2a9.png)

### Intermediate training data (for route navigator)

- crevasse_map.tif (predicted mask) - seamless geotiff output of an entire area from [crevasse finder](https://github.com/weiji14/nz_space_challenge/blob/master/crevasse_finder.ipynb)
- earthobservation_map.tif (image) - the cropped geotiff satellite earth observation image used as input to produce the crevasse_map.tif file above.

![Crevasse map of Ross Ice Shelf](https://user-images.githubusercontent.com/23487320/39153891-9b7c3856-47a0-11e8-8fbe-0ba969045d50.png)

