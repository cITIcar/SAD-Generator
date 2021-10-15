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
TODO: full explanation probably in another md file

### chunks
For synthetic image creation the `chunk` folder is used to dynamically create and annotate a ground plane 
from the funamental building blocks `curve_left`, `curve_right`, `line` and `intersection`.  
The additional JSON files define the driving path of the simulated vehicle along the road.
Each of the chunk type may have several variants that use the same segmentation map. While running a random variant
chosen for each new segment.

### white_box
TODO


### render_objects
TODO
