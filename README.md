# [New Zealand Space Challenge 2018](https://www.nzspacechallenge.com/)
Detecting crevasses in Antarctica for safer, more efficient navigation as an analogue for future space missions.

Experimental (alpha) leaflet map demo using tensorflowjs [here](https://weiji14.github.io/nz_space_challenge/).

Youtube video giving a quick overview explanation [here](https://www.youtube.com/watch?v=Vy-f852grFg).

# CrevasseNet model architecture

Consists of a [classifier module](./crevasse_finder.ipynb) seamlessly joined to a [navigator module](./route_finder.ipynb), trained using supervised learning and reinforcement learning respectively.

![model_architecture](https://user-images.githubusercontent.com/23487320/39678382-dd2066bc-51df-11e8-99e0-7d88299a3146.png)

Note that the classifier component is actually much deeper, but has been abbreviated in the above diagram for simplicity.

## Sample predictions

### [Crevasse Classifier](./crevasse_finder.ipynb)

Input image (satellite/aerial)--> Intermediate Output (crevasse map)

![crevasse_prediction](https://user-images.githubusercontent.com/23487320/39678490-e5883350-51e1-11e8-8483-60fbb84865d9.png)

### [Route Navigator](./route_finder.ipynb)

Intermediate output (crevasse map) --> Action quality outputs

![route_navigator.gif](https://user-images.githubusercontent.com/23487320/39678342-fffe1798-51de-11e8-9061-0e7cc94f32e3.gif)

# Getting started

## Quickstart

Launch Binder, data will be loaded via [Quilt](https://github.com/quiltdata/quilt). Cheers to [data2binder](https://github.com/quiltdata/data2binder)!

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/weiji14/nz_space_challenge/master)

## Installation

Start by cloning this [repo-url](/../../)

    git clone <repo-url>
    cd nz_space_challenge
    conda env create -f environment.yml

## Running the jupyter notebook

    source activate nz_space_challenge
    python -m ipykernel install --user  #to install conda env properly
    jupyter kernelspec list --json      #see if kernel is installed
    jupyter notebook


# [Data used](/data)

| Name                                                                 | Data Source                                      |
| -------------------------------------------------------------------- | ------------------------------------------------:|
|MOA-derived Structural Feature Map of the Ronne Ice Shelf, Version 1  | [NSIDC-0497](https://nsidc.org/data/nsidc-0497)  |
|MODIS Mosaic of Antarctica 2003-2004 (MOA2004) Image Map, Version 1   | [NSIDC-0280](https://nsidc.org/data/nsidc-0280)  |

