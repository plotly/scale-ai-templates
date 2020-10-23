## About the app
The videos are displayed using a community-maintained Dash video component. It is made by two Plotly community contributors. You can find the [source code here](https://github.com/SkyRatInd/Video-Engine-Dash).

All videos used are open-sourced under Creative Commons. The [original links can be found here](data/original_footage.md).

### Model
The object detection model is the MobileNet v1, made by Google and trained on the COCO dataset. You can find their implementation on their [official Github repo](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md). You are encouraged to try this app with other models.

### Bounding Box Generation
The data displayed in the app are pregenerated for demo purposes. To generate the csv files containing the objects detected for each frame, as well as the output video with bounding boxes, please refer to `utils/generate_video_data.py`. You will need the latest version of tensorflow and OpenCV, as well as the frozen graph `ssd_mobilenet_v1_coco`, that you can [download in the Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). Make sure to place the frozen graph inside the same folder as `generate_video_data.py`, i.e. `utils`.

### Difference between the python version and the R version
The R version has 3 modes instead of two modes: (Visual Mode & Confidence Mode & Detection Mode) in lieu of 2 modes (Visual Mode & Detection Mode).

The heatmap component now has a separate mode with two columns of labels written below the graph. It was a sacrifice in terms of design aspect at the benefit of greater performance and stability.