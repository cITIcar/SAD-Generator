## JSON Configuration

### Camera Options

The Camera Options are used to place a virtual camera into the scene.

- `fov`: horizontal field of view in degrees, e.g. `90`
- `camera_height`: distance to the ground from the camera
- `camera_angle`: pitch angle of the camera in degrees
- `output_size`: width and height in pixels of the output camera image
- `rescale`: upscale factor that increases the size of the rendered image to preserve lines in the distance
- `chunk_size_cm`: size of the chunks in the `chunk` folder in cm
- `chunk_size_px`: size of the chunks in the `chunk` folder in pixels 

- `shuffle`: shuffle the output images
- `seed`: seed to initialize the random generator

### Path Options
- `images_output_path`: Under this path the augmented and synthetic images will be stored, e.g. `data/images/{splitname}/some_set`. For the creation of separated datasets, the path should contain the variable `{splitname}`, otherwise all splits will be saved in the same folder.
- `labels_output_path`: Under this path the generated labels of the augmented and synthetic images will be stored, e.g. `data/labels/{splitname}/some_set`. A label belongs to a image if they have the same name.
- `manual_images_input_path`: This is the path to the images of the manually annotated data.
- `manual_labels_input_path`: This is the path to the labels of the manually annotated data. A label belongs to a image if they have the same name.
- `chunk_path`: This is the path to the chunk folder. 
- `chunk_file_pattern`: This describes how the chunks will be named. Every chunk image in the chunk folder has a unique name with the suffix `.png`. Every chunk label in the chunk folder has a unique name with the suffix `_label.png`. Two variables need to be defined inside the pattern: The `chunk_type` (e.g. `line`) and the `variant` (There may be multiple variants for one chunk type).

### Dataset Options 
The config key `splits` which datasets will be outputed from the data generator.  Each `split` contains the total number of images as well as the fraction of which should be fully synthetic or augmented. In this example we defined a train dataset `train_split`, a validation dataset `validation_split` and a test dataset `test_split`.

Example
```json

"splits": {
    "train_split": {
        "size": 10000,
        "fraction_synthetic": 1.0,
        "fraction_augmented": 0
    },
    "validation_split": {
        "size": 300,
        "fraction_synthetic": 0.5,
        "fraction_augmented": 0.5
    },
    "test_split": {
        "size": 300,
        "fraction_synthetic": 0,
        "fraction_augmented": 1.0
    }
},

```

### Disturbances

`disturbances` should contain a list of of each disturbance that is to be added to the generator where the key represent the name of the class in question and the value of each key is the parameter list with which it is initialized.  
Note that in addition to these options the global configuration is passed to each disturbance instance as well. 

- `HiddenGroundRect`: 
- `BoxObstacle`: 
- `Dust`: 
- `RandomBrightness`: 
- `DrunkDriving`: 

