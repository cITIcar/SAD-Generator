import numpy as np
import math
import json


class Config:
    """This class contains the configuration.

    For convenience it also adds a few keys,that can be immediately created
    from the given configuration file.

    Attributes
    ----------
    config : dict
        The configuration file
    """

    def __init__(self, config_file, debug=False):
        """Creates a new instance.

        Parameters
        ----------
        config_file : str
            Path to a config file.
        """

        self.config = json.load(open(config_file))

        self.config["intrinsic_camera_matrix"] = Config.create_intrinsic_from(
            self.config["fov"],
            [self.config["output_size"][0] * self.config["rescale"],
             self.config["output_size"][1] * self.config["rescale"]])

        self.config["intrinsic_segmentation_matrix"] = Config.create_intrinsic_from(
            self.config["fov"],
            self.config["output_size"])

        self.config["project2d"] = Config.create_project2d()

        self.config["px_per_cm"] = self.config["chunk_size_px"] / self.config["chunk_size_cm"]
        self.config["camera_height_px"] = self.config["camera_height"] * self.config["px_per_cm"]
        self.config["debug"] = debug

    def __getitem__(self, key):
        """Imlpements Pythons index operation for ease of use.

        Parameters
        ----------
        key : str
            The name of the item.

        Returns
        -------
            The item.
        """
        return self.config[key]

    def create_intrinsic_from(fov, output_size):
        """Create the intrinsic camera matrix.

        Parameters
        ----------
        fov : int
            Field of view of the camera in degrees.

        Returns
        -------
            The intrinsic camera matrix
        """
        output_width, output_height = output_size
        focal_length = (output_width / 2) / math.tan((fov / 180 * np.pi) / 2)

        return np.float32([
            [-focal_length, 0, output_width / 2, 0],
            [0, -focal_length, output_height / 2, 0],
            [0, 0, 1, 0]])

    def create_project2d():
        """Create a matrix to project 3D homogenous coordinates back into 2D.

        Returns
        -------
            The project 2D matrix.
        """
        return np.float32([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 0],
            [0, 0, 1]])
