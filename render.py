"""Contains functions to implement perspective transforms."""

import numpy as np
from numpy import sin, cos
import cv2


class Renderer:
    """This class is used to generate Synthetic images from the cameras perspective.

    Attributes
    ----------
    config : Config
        The current configuration.
    angle_x : float
        The angle around the x axis.
    angle_y : float
        The angle around the y axis.
    angle_z : float
        The angle around the z axis.
    position_x : int
        The camera position on the x axis.
    position_y : int
        The camera position on the y axis.
    position_z : int
        The camera position on the z axis.
    render_objects : Disturbance
        A list of Disturbances that are to be added.
    horizon_fraction : float
        The fraction of the image height, where the horizon line is.
    """

    def __init__(self, config):
        self.config = config
        self.angle_x = np.pi / 2 + np.radians(config["camera_angle"])
        self.angle_y = np.radians(0)
        self.angle_z = np.radians(0)

        self.position_x = 0
        self.position_y = 0
        self.position_z = config["camera_height_px"]

        self.render_objects = []
        self.update_position((self.position_x, self.position_y), self.angle_z)
        horizon_point = self.h_label @ np.array([[0], [1], [0]])
        self.horizon_fraction = ((horizon_point[1] / horizon_point[2]) /
                                 self.config["output_size"][1])

    def project_point(self, coord):
        """Projects a single 3D point into the current camera perspective.

        Parameters
        ----------
        coord : np.ndarray
            A 3D position with x, y, z.

        Returns
        -------
        coord_2d : np.ndarray
            The position of coord in the camera's perspective.

        """
        x, y, z = coord
        coord_3d_h = np.array([-x, -y, -z, 1])
        coord_3d_h = self.translation @ coord_3d_h
        coord_2d_h = (self.config["intrinsic_camera_matrix"] @
                      self.rotation @ coord_3d_h)
        coord_2d = coord_2d_h[:2] / coord_2d_h[2]

        return coord_2d

    def update_position(self, position, angle):
        """Updates the position and yaw angle of the camera.

        Parameters
        ----------
        position : list[int]
            The x and y position of the camera.
        angle : float
            The yaw angle of the camera.

        Returns
        -------
        None.

        """
        for obj in self.render_objects:
            if obj.update_position_step:
                position, angle = obj.update_position_step(
                    position=position,
                    angle=angle,
                    renderer=self)

        self.position_x, self.position_y = position
        self.angle_z = -angle
        rotation_x = np.array([
            [1, 0, 0, 0],
            [0, cos(self.angle_x), -sin(self.angle_x), 0],
            [0, sin(self.angle_x), cos(self.angle_x), 0],
            [0, 0, 0, 1]])
        rotation_y = np.array([
            [cos(self.angle_y), 0, sin(self.angle_y), 0],
            [0, 1, 0, 0],
            [-sin(self.angle_y), 0, cos(self.angle_y), 0],
            [0, 0, 0, 1]])
        rotation_z = np.array([
            [cos(self.angle_z), -sin(self.angle_z), 0, 0],
            [sin(self.angle_z), cos(self.angle_z), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
        self.rotation = rotation_x @ rotation_y @ rotation_z

        self.translation = np.array([
            [1, 0, 0, self.position_x],
            [0, 1, 0, self.position_y],
            [0, 0, 1, self.position_z],
            [0, 0, 0, 1]])

        self.h_camera = (self.config["intrinsic_camera_matrix"] @
                         self.rotation @ self.translation @
                         self.config["project2d"])
        self.h_label = (self.config["intrinsic_label_matrix"] @
                               self.rotation @ self.translation @
                               self.config["project2d"])

    def update_ground_plane(self, image_real, image_label,
                            render_objects):
        """Updates the current scene.

        Parameters
        ----------
        image_real : np.ndarray
            Birds-eye-view of the ground.
        image_label : np.ndarray
            Birds-eye-view of the label map.
        render_objects : List[Disturbance]
            A list of the disturbances that are to be rendererd.

        Returns
        -------
        None.

        """
        self.image_real = image_real
        self.image_label = image_label
        self.render_objects = sorted(render_objects,
                                     key=lambda obj: obj.ordering)

        for obj in self.render_objects:
            if obj.pre_transform_step:
                obj.pre_transform_step(
                    image=image_real,
                    image_segment=image_label)

        self.render_objects.sort(key=lambda obj: obj.ordering)

    def render_images(self):
        """Render the ground plane and disturbances from camera perspective.

        Parameters
        ----------
        None.

        Returns
        -------
        perspective_camera : np.ndarray
            The normal output image.
        perspective_label : TYPE
            The image's label map.

        """
        width, height = self.config["output_size"]
        rescale = self.config["rescale"]
        perspective_camera = cv2.warpPerspective(
            self.image_real, self.h_camera,
            (width * rescale, height * rescale), flags=cv2.INTER_NEAREST)
        perspective_label = cv2.warpPerspective(
            self.image_label, self.h_label,
            (width, height), flags=cv2.INTER_NEAREST)

        perspective_camera[:int(len(perspective_camera) *
                                self.horizon_fraction)] = 0
        perspective_label[:int(len(perspective_label) *
                               self.horizon_fraction)] = 0

        for obj in self.render_objects:
            if obj.post_transform_step:
                obj.post_transform_step(
                    image=perspective_camera,
                    image_segment=perspective_label,
                    point=(self.position_x, self.position_y),
                    angle=self.angle_z,
                    global_angle=self.angle_z,
                    bird_to_camera_nice=self.h_camera,
                    bird_to_camera_segment=self.h_label,
                    renderer=self)

        if rescale != 1:
            perspective_camera = cv2.resize(
                perspective_camera, (width, height),
                interpolation=cv2.INTER_AREA)

        return perspective_camera, perspective_label
