
# Data Generator

![example video](example.webm)

Robust scene understanding algorithms are essential for the success of autonomous navigation. Unfortunately the supervised learning of semantic segmentation requires large and diverse datasets. For certain self-driving tasks like navigating a robot inside an industrial facility no datasets are freely available and the manual annotation of large datasets is impracticable for smaller development teams. Although approaches exist to automatically generate synthetic data, they are either too computational expensive, demand a huge preparation effort or miss a large variety of features.  

This data generator gives a simple and fast approach to create artificial datasets with sufficient variety for self-driving tasks on flat ground planes.

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
from the funamental building blocks `curve_left`, `curve_right`, `line` and `intersection`.
The additional JSON files define the driving path of the simulated vehicle along the road.
Each of the chunk type may have several variants that use the same segmentation map. While running a random variant
chosen for each new segment.

### white_box
TODO


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