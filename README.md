# PYDRM - A graphical user interface for DRM data analysis

Find more information about this project [here](https://mallorywittwer.github.io/post_article/4.html)!

This version includes a **grain segmentation panel** which lets the user compute grain statistics by click-and-dragging a region of interest on the canvas. The segmentation algorithm is designed to work without any parameter-tuning. Details about the implementation of the grain segmentation algorithm can be found in [our publication](https://doi.org/10.1016/j.matchar.2021.110978). The number of NMF components for the decomposition and sampling size can be adjusted for optimal results.

![segment_im_demo](https://user-images.githubusercontent.com/39482871/118356204-aa391b80-b5a6-11eb-86be-5a1076f53e31.jpg)

## Installation

We recommend installing Python > 3.6 and the dependencies in `requirements.txt` in a fresh environment. The app can be started by running `app.py`.
