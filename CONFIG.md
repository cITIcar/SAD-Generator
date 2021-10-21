## JSON Configuration

### Camera Options

The Camera Options are used to place a virtual camera into the scene.

- `fov`: Horizontal field of view in degrees, e.g. `90`.
- `camera_height`: Distance to the ground from the camera.
- `camera_angle`: Pitch angle of the camera in degrees
- `output_size`: width and height in pixels of the output camera image
- `rescale`: upscale factor that increases the size of the rendered image to preserve lines in the distance
- `chunk_size_cm`: size of the chunks in the `chunk` folder in cm
- `chunk_size_px`: size of the chunks in the `chunk` folder in pixels 

### Path Options

- `annotations_output_path`: The output path for both the augmented and synthetic segmentation images. The path should contain the variable `{splitname}` somewhere, otherwise all splits will be saved in the same folder, e.g. `data/annotations/{splitname}/some_set`.
- `images_output_path`: Same as above but for the camera images.
- `manual_annotations_input_path`: Path to the manually annotated segmentation images.
- `manual_images_input_path`: Path to the original manually annotated images.
- `chunk_path`: Path to the chunk folder that contains the ground chunks to put together.
- `chunk_file_patter`: a pattern identifying a chunk image. The actual image file used is then the pattern + `.png` or pattern + `_segment.png` for the image files or segmentation files respectively.  
Two variables need to be defined inside the pattern: The `chunk_type` and the `variant`.

### Dataset Options 
The config key `splits` contains information about each data split.  
In it a `train_split`, `validation_split` and `test_split` need to be defined.

Each `split` contains the total number of images as well as the fraction of which should be fully synthetic or augmented.

An example split configuration is:
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
Note that in addition to these options the globa configuration is passed to each disturbance instance as well. 

### Other Options

- `shuffle`: shuffle the output images
- `seed`: seed to initialize the random generators

The config file is also automatically forwarded to any `Disturbance` objects, so that further configuration options may be added for that purpose.

### Example Configuration	

```json
{
    "fov": 90,
    "camera_height": 25.5,
    "camera_angle": -5,
    "output_size": [128, 64],
    "rescale": 8,
    "chunk_size_cm": 250,
    "chunk_size_px": 1000,
    "shuffle": true,
    "seed": 123456,

    "obstacle_class": 200,

    "paths": {
        "annotations_output_path": "data/annotations/{splitname}/some_set",
        "images_output_path": "data/images/{splitname}/some_set",
        "manual_annotations_input_path": "real_dataset/annotations",
        "manual_images_input_path": "real_dataset/images",
        "chunk_path": "chunks",
        "chunk_file_pattern": "{chunk_type}_{variant}",
        "output_file_pattern": "image_{idx}.png"
    },

    "splits": {
        "train_split": {
            "size": 1000,
            "fraction_synthetic": 0.5,
            "fraction_augmented": 0.5
        },
        "validation_split": {
            "size": 300,
            "fraction_synthetic": 0.5,
            "fraction_augmented": 0.5
        },
        "test_split": {
            "size": 300,
            "fraction_synthetic": 0.5,
            "fraction_augmented": 0.5
        }
    },

    "disturbances": {
        "HiddenGroundRect": [],
        "BoxObstacle": [],
        "BoxObstacle": [],
        "Dust": [100000, 63],
        "Dust": [100000, 127],
        "Dust": [10000, 255],
        "RandomBrightness": [],
        "RandomClipMax": [],
        "RandomClipMax": [],
        "DrunkDriving": []
    }
}
```