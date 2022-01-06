# PYDRM - A graphical user interface for DRM data analysis

This version includes a **grain segmentation panel** which lets the user compute grain statistics by click-and-dragging a region of interest on the canvas. The segmentation algorithm is designed to work without any parameter-tuning. Details about the implementation of the grain segmentation algorithm can be found in [our publication](https://doi.org/10.1016/j.matchar.2021.110978). The number of NMF components for the decomposition and sampling size can be adjusted for optimal results.

![segment_im_demo](https://user-images.githubusercontent.com/39482871/118356204-aa391b80-b5a6-11eb-86be-5a1076f53e31.jpg)

## Installation

We recommend installing Python > 3.6 and the dependencies listed in `requirements.txt` in a fresh environment. Alternatively, one can execute the code in the provided virtual environement **environment.yml** using [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) (to install Anaconda: see [here](https://www.anaconda.com/)). To create the environement, use:

`conda env create -f environment.yml`

This will create an environment with the name *drm_ml*. To activate that environment, use:

`conda activate pydrm_GUI-env`

We tested the code using the following dependencies:

- python 3.8.10
- numpy 1.19.5
- pandas 1.3.1
- matplotlib 3.4.2
- scikit-image 0.18.2
- scikit-learn 0.24.2
- tensorflow 2.5.0
- psutil 5.8.0

## Execution

The app can be started by running `app.py`.
