# PYDRM - A GUI for DRM data analysis

Find information about this project [here](https://mallorywittwer.github.io/post_article/4.html)!

This version of the GUI includes a **grain segmentation panel** which lets the user compute grain statistics seamlessly, by simply click-and-dragging a region of interest on the canvas. The segmentation algorithm is designed to work without any parameter-tuning. Details about its implementation can be found [here](https://www.sciencedirect.com/science/article/pii/S104458032100108X?via%3Dihub) (full publication) or in this [short review article](https://mallorywittwer.github.io/post_article/2.html). We still give the option to choose the number of NMF components for the decomposition and to reduce the sampling size in order to speed up the calculation.

![segment_im_demo](https://user-images.githubusercontent.com/39482871/118356204-aa391b80-b5a6-11eb-86be-5a1076f53e31.jpg)
