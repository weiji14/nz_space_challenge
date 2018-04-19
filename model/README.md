# Crevasse training data for [NZ Space Challenge](https://www.nzspacechallenge.com/)

This folder contains the Keras/Tensorflow training datasets for the github repository found [here](https://github.com/weiji14/nz_space_challenge).
This training dataset will be used to perform a supervised classification algorithm to detect crevasse lines in Antarctica from satellite imagery.

The dataset is prepared from this [data preparation](https://github.com/weiji14/nz_space_challenge/blob/master/data_prep.ipynb) jupyter notebook, and it will feed into the [crevasse finder](https://github.com/weiji14/nz_space_challenge/blob/master/crevasse_finder.ipynb) Convolutional Neural Network model.

Click on the [binder](https://mybinder.org) link button below to launch an interactive notebook!
The data will be loaded via [Quilt](https://github.com/quiltdata/quilt), using instructions from [data2binder](https://github.com/quiltdata/data2binder).

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/weiji14/nz_space_challenge/master)


## Description of dataset

### Training data

- X_data.npy (image) - from the MODIS Mosaic of Antarctica 2003-2004 (MOA2004) Image Map, Version 1 [NSIDC-0280](https://nsidc.org/data/nsidc-0280)
- Y_data.npy (mask) - from the MOA-derived Structural Feature Map of the Ronne Ice Shelf, Version 1 [NSIDC-0497](https://nsidc.org/data/nsidc-0497)

![MODIS image and crevasse lines](https://user-images.githubusercontent.com/23487320/38399063-23bc975a-399c-11e8-8440-54cd412489dd.png)

### Intermediate output data

- W_hat_data.npy (predicted crevasse mask) - seamless output of an entire area from [crevasse finder](https://github.com/weiji14/nz_space_challenge/blob/master/crevasse_finder.ipynb)
- crevasse_map.tif - a geotiff version of the above crevasse mask

![Crevasse map of Ross Ice Shelf](https://user-images.githubusercontent.com/23487320/38966422-492b6432-43d6-11e8-9a9f-99e925124c89.png)

