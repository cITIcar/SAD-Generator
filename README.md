
# Data Generator

This repository contains code to easily generate synthetic datasets or augment manually annotated real world datasets.

https://user-images.githubusercontent.com/88937076/138556367-ae4f4292-7c83-414f-9d0e-bb263280cec3.mp4

Robust scene understanding algorithms are essential for the success of autonomous navigation. Unfortunately the supervised learning of semantic segmentation requires large and diverse datasets. For certain self-driving tasks like navigating a robot inside an industrial facility no datasets are freely available and the manual annotation of large datasets is impracticable for smaller development teams. Although approaches exist to automatically generate synthetic data, they are either too computational expensive, demand a huge preparation effort or miss a large variety of features.  

This data generator gives a simple and fast approach to create artificial datasets with sufficient variety for self-driving tasks on flat ground planes.

## Citing

If our code helped you, please consider citing our paper:  
- [1] Pau Dietz Romero, Merlin David Mengel, Jakob Czekansky  
  [Synthesising large, low cost and diverse datasets for robust semantic segmentation in self-driving tasks]()


## Creating a virtual environment

It is recommended to create a [virtual environment](https://docs.python.org/3/library/venv.html#venv-def) and install all dependencies inside:
```
# under Linux
python3 -m venv venv4datagenerator
source venv4datagenerator/bin/activate

# under Windows
# the PATH variable needs to be configured, otherwise use the full path to the python executable
python -m venv venv4datagenerator
venv4datagenerator\Scripts\activate.bat
```

## Dependencies

- OpenCV
- Numpy

Run
```
pip install -r requirements.txt
```
to install dependencies.

## Running

Use
```bash
python main.py --config config1.json
```
to create a new dataset using default config file.

For debugging purposes the switch `--debug` may be used to display the video stream only, instead of saving the image files.

## Configuration

### JSON Config File
JSON configuration files are used for most of the configuration.

The full reference is available in [CONFIG.md](CONFIG.md).

### chunks
For synthetic image creation the `chunk` folder is used to dynamically create and annotate a ground plane
from the fundamental building blocks `curve_left`, `curve_right`, `line` and `intersection`.
The additional JSON files define the driving path of the simulated vehicle along the road.
Each of the chunk type may have several variants that use the same segmentation map. While running a random variant
chosen for each new segment.

This figure shows how the ground plane is modularly constructed by chunks.
<img width="50%" src="https://user-images.githubusercontent.com/88937076/138558644-222a3bcd-ea1d-46ec-918f-5033dd79b3ef.png"></img>

In our example the chunks represent the building blocks of a miniature track for a model car. By changing the chunk images, one can adapt the data generator to their own self-driving task. Every chunk has its annotation. 
The chunk images need a consistent scale, size which has to be defined in the [config](CONFIG.md). Additionally the images have to be undistorted and in the bird's-eye-view.

### overlays

Images in this folder are randomly added to the camera images to enhance their variety. They serve as disturbing artefacts and do not affect the annotations of the images.   
Further disurbing images may be inserted here.

### white_box

Images in this folder are added to the camera images and the annotations using the `obstacle_class` defined in the config file.  
They represent objects that have to be classified by the network.  
In our example the objects are obstacles.

### disturbances
The generator is capable of creating various disturbing factors.
These may be objects that are added to the scene, or changes in the driving behaviour.

The folder `disturbances` contains a set of predefined disturbing factors:

- _BoxObstacle_ Places a box of random size and position into the scene.
Since there is a class associated with an obstacle, it is rendered both in the camera image and the segmentation image.
- _RandomClipMax_ Reguarily Chooses a random frequency at which it clips the brightness of the output image.
- _RandomBrightness_ Randomly brightens the output image.
- _HiddenGroundRect_ Hides a section of the ground. The segmentation image is no affected.
- _Dust_ Sets a number of randomly selected pixels on the ground plane to a defined color to simulate dust.
- _DrunkDriving_ Adds a continuous error following a sine wave to the drive path to simulate mistakes in the driving behavior.

#### Adding a new Disturbance

Each disturbance needs to inherit the class `Disturbance` defined in `disturbances/disturbance.py`.  
A new class may then overwrite one or more of the methods `pre_transform_step`, `post_transform_step` and `update_position_step`.  

The `pre_transform_step` is applied whenever the ground plane is updated in the renderer.  
In this step a reference to the global ground plane both segmented and real is passed to the method. 

`post_transform_step` is applied after the ground plane has been rendered into the image. This step may be used to apply effects to the camera image, like changes in contrast or brightness or to render in objects using `renderer.project_point`.  
The parameters passed to this method are:

- `image`: The camera image.
- `image_segment`: The segmentation image.
- `point`: The current position of the vehicle.
- `angle` The current angle of the vehicle relative to the current ground plane.
- `bird_to_camera_nice`: The homography matrix to transform the ground plane according to the current position for the camera. 
- `bird_to_camera_segment`: The homography matrix to transform the ground plane for the segmentation image.
- `renderer`: A reference to the renderer object.

The `update_position_step` is called whenever the position of the vehicle is updated. It receives the current position and angle and is expected to return both as a tuple, making it possible to modify the drive path defined in the chunk JSON files.

Finally, the class field `ordering` is used to determine the rendering order. It is used both for the order of the `pre_transform_step` and the `post_transform_step`, but can be modified in between. For rendering objects this value should be set in the `pre_transform_step` to the distance of the object to the camera.
