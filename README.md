# Data Generator

This data generator is meant to create datasets for training image segmentation for self-driving vehicles using a combination of fully synthetic and manually annotated images.

## Dependencies

TODO: requirements.txt

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

The folder `disturbances` contains a set of predefined desturbing factors:

- _BoxObstacle_ places a box of random size and position into the scene.
Since there is a class associated with an obstacle, it is rendered both in the camera image and the segmentation image.
- _RandomClipMax_ reguarily chooses a random frequency at which it clips the brightness of the output image.
- _RandomBrightness_ randomly brightens the output image.
- _HiddenGroundRect_ hides a section of the ground. The segmentation image is no affected.
- _Dust_ sets a number of randomly selected pixels on the ground plane to a defined color to simulate dust.
- _DrunkDriving_ adds a continuous error following a sine wave to the drive path to simulate mistakes in the driving behavior.
