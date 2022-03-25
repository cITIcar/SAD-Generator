
# SAD - Synthetic Augmented Data Generator

This repository contains code to easily generate synthetic datasets or augment manually annotated real world datasets.

https://user-images.githubusercontent.com/88937076/138556367-ae4f4292-7c83-414f-9d0e-bb263280cec3.mp4

Robust scene understanding algorithms are essential for the success of autonomous navigation. Unfortunately the supervised learning of semantic segmentation requires large and diverse datasets. For certain self-driving tasks like navigating a robot inside an industrial facility no datasets are freely available and the manual annotation of large datasets is impracticable for smaller development teams. Although approaches exist to automatically generate synthetic data, they are either too computational expensive, demand a huge preparation effort or miss a large variety of features.  

This data generator gives a simple and fast approach to create artificial datasets with sufficient variety for self-driving tasks on flat ground planes.

## Citing

If our code helped you, please consider citing our paper:  
- [1] Pau Dietz Romero, Merlin David Mengel, Jakob Czekansky  
  [Synthesizing large, low cost and diverse datasets for robust semantic segmentation in self-driving tasks]()


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

- opencv-python
- numpy
- imutils
- jupyterlab
- matplotlib

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

The data generator picks the basic building blocks (chunks) from the folder `chunks` to dynamically create and annotate a ground plane. There are four different chunk types: `curve_left`, `curve_right`, `line` and `intersection`. For every chunk type there has to exist exactly one label and at least one image. If multiple images exist for one type they would be called variants of the chunk. The label contains the semantic information of the chunk. The image is the visual appearance of the road element. Every chunk type has a correspondent JSON file with its metadata. One important meta information is the ideal path that the car would take through the chunk.

This figure shows how the ground plane is modularily constructed from chunks.
<img width="50%" src="https://user-images.githubusercontent.com/88937076/138558644-222a3bcd-ea1d-46ec-918f-5033dd79b3ef.png"></img>

In our example the chunks represent the building blocks of a miniature track for a model car. By changing the chunk images, one can adapt the data generator to their own self-driving task. The chunk images need a consistent scale and size which has to be defined in the [config](CONFIG.md). Additionally the images have to be in the bird's-eye-view.

### overlays

Overlays are random distracting images which the data generator puts on the camera image to enhance the variety of the dataset. They serve as disturbing artifacts and do not affect the annotations of the images because they do not belong to a segmentation class. The user can add their own overlays inside the folder `overlays`.

### white_box

The images in this folder are added to the camera images and the annotations using the `obstacle_class` defined in the config file. Compared to the overlays, they do affect the annotations of the images because they do belong to a segmentation class. Therefore they represent objects that have to be classified by the network.  In our example the objects are obstacles.

### disturbances
The generator is capable of creating various disturbing factors.
These may be objects that are added to the scene, or changes in the driving behavior.

The folder `disturbances` contains a set of predefined disturbing factors:

- _BoxObstacle_ Places a box of random size and position into the scene.
Since there is a class associated with an obstacle, it is rendered both in the camera image and the segmentation image.
- _RandomClipMax_ Regularly Chooses a random frequency at which it clips the brightness of the output image.
- _RandomBrightness_ Randomly brightens the output image.
- _HiddenGroundRect_ Hides a section of the ground. The segmentation image is no affected.
- _Dust_ Sets a number of randomly selected pixels on the ground plane to a defined color to simulate dust.
- _DrunkDriving_ Adds a continuous error following a sine wave to the drive path to simulate mistakes in the driving behavior.

#### Adding a new Disturbance

Each disturbance needs to inherit from the class `Disturbance` defined in `disturbances/disturbance.py`.  
A new class may then overwrite one or more of the methods `pre_transform_step`, `post_transform_step` and `update_position_step`.  

The `pre_transform_step` is applied whenever the ground plane is updated in the renderer.  
In this step a reference to the global ground plane both segmented and real is passed to the method. 

`post_transform_step` is applied after the ground plane has been rendered into the image. This step may be used to apply effects to the camera image, like changes in contrast or brightness or to render in objects using `renderer.project_point`.  
The parameters passed to this method are:

- `image`: the camera image
- `image_segment`: the segmentation image
- `point`: the current position of the vehicle
- `angle` the current angle of the vehicle relative to the current ground plane
- `bird_to_camera_nice`: the homography matrix to transform the ground plane according to the current position for the camera
- `bird_to_camera_segment`: the homography matrix to transform the ground plane for the segmentation image
- `renderer`: a reference to the renderer object

The `update_position_step` is called whenever the position of the vehicle is updated. It receives the current position and angle of the car and is expected to return both as a tuple, making it possible to modify the drive path defined in the chunk JSON files.

Finally, the class field `ordering` is used to determine the rendering order. It is used both for the order of the `pre_transform_step` and the `post_transform_step`, but can be modified in between. For rendering objects this value should be set in the `pre_transform_step` to the distance of the object to the camera.

### Manual Augmentation

https://user-images.githubusercontent.com/88937076/157739977-2eee7219-2c1b-4ba0-9416-3d1052bb2a5c.mp4

This repository offers an interactive GUI for manual augmentation.

The class `ManualAugment` in the file `manual_augment.py` contains generic functions for manually adding overlays to annotated samples.
The program `startline.py` shows an example of the manual augmentation. Here a start line will be added to the image and annotation of the data sample.
A interactive GUI is offered to the user where he can translate and rotate the overlay inside the image.
When the overlay has reached the final pose, the user can save the image. The overlay will be automatically added to the image and annotation.

Use
```bash
python startline.py
```
to test the example.

The start line can be moved using the following controls:

| Key | Action |
|-----|--------|
| W   | move start line away from the camera  | 
| A   | move start line to the left | 
| S   | move start line towards the camera| 
| D   | move start line to the right | 
| E   | rotate the start line clockwise | 
| R   | rotate the start line counter-clockwise | 
| Q   | skip the image | 
| Space   | save the image and go to the next | 
| X   | quit | 

Keep in mind that the input and output paths for the images have to match.
The paths are set in the configuration file `config1.json`.
