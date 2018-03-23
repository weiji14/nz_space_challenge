# [New Zealand Space Challenge 2018](https://www.nzspacechallenge.com/)
Detecting crevasses in Antarctica for safer, more efficient navigation as an analogue for future space missions

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
