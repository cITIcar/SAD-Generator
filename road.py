"""Contains functions to create the background image of a road."""

import cv2 as cv
import numpy as np
import json
import glob


class Road:
    """
    Objects of this class represent a large street map.

    Attributes
    ----------
    images : Dict
        Data structure to store all images and annotations of chunks.
    degree_list : List
        List with angle information for each previous chunk.
    size_image_px : int
        Size of chunk in number of pixels.
    file_list : List
        Names of all chunks of the previous road.
    """

    def __init__(self, config):
        self.images = {}
        self.chunk_json = {}
        self.file_list = ["line", "line", "intersection", "line"]
        self.degree_list = [0, 0, 0, 0]
        self.size_image_px = 1000
        self.config = config
        self.load_images()

    def load_images(self):
        """
        Load all chunks into RAM.

        This function saves computing time during the simulation. Instead of
        loading the images from the persistent memory each time and performing
        the necessary rotations, all possible rotations are stored in a quickly
        accessible list.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        for segment_type in ["line", "intersection", "curve_left",
                             "curve_right"]:
            self.images[segment_type] = {"segment": [], "nice": []}
            for variant in sorted(glob.glob(
                    f"chunks/{segment_type}_segment*.png")):
                segment = cv.imread(variant, cv.IMREAD_GRAYSCALE)

                self.images[segment_type]["segment"].append({
                    0: segment.astype(np.float32),
                    90: cv.rotate(segment, cv.ROTATE_90_COUNTERCLOCKWISE).astype(np.float32),
                    -180: cv.rotate(segment, cv.ROTATE_180).astype(np.float32),
                    180: cv.rotate(segment, cv.ROTATE_180).astype(np.float32),
                    -90: cv.rotate(segment, cv.ROTATE_90_CLOCKWISE).astype(np.float32),
                })

            with open(f"chunks/{segment_type}.json") as f:
                self.chunk_json[segment_type] = json.load(f)

            path_pattern = (
                self.config["paths"]["chunk_path"] + "/" +
                self.config["paths"]["chunk_file_pattern"]
            ).format(chunk_type=segment_type + "_nice", variant="*")

            for variant in sorted(glob.glob(path_pattern)):
                nice = cv.imread(variant, cv.IMREAD_GRAYSCALE)

                self.images[segment_type]["nice"].append({
                    0: nice.astype(np.float32),
                    90: cv.rotate(nice, cv.ROTATE_90_COUNTERCLOCKWISE).astype(np.float32),
                    -180: cv.rotate(nice, cv.ROTATE_180).astype(np.float32),
                    180: cv.rotate(nice, cv.ROTATE_180).astype(np.float32),
                    -90: cv.rotate(nice, cv.ROTATE_90_CLOCKWISE).astype(np.float32),
                })

    def get_position(self):
        """
        Get the position of the next chunk.

        THe position of the next chunk depends on the angle of the previous
        chunk in the mn coordinate system.

        Parameters
        ----------
        None.

        Returns
        -------
        position_list : list
            List with positions of all chunks in the mn coordinate system
        total_degree_list : list
            List with resulting angle information of all chunks
        """
        # Set the first chunk to the origin of the mn coordinate system
        position_list = [(0, 0)]

        # The first chunk starts always with angle zero.
        total_degree_list = [0]

        # Set the second to last chunk.
        for i in range(1, len(self.degree_list)):

            # Sum all previous angles to determine total rotation
            total_degree = sum(self.degree_list[1:i+1])
            total_degree_list.append(total_degree)

            # Get the position of the previous chunk
            (m, n) = position_list[i-1]

            # previous chunk points right -> next chunk right
            if total_degree_list[i] == 90:
                m = m + 1
            # previous chunk points to the left -> next chunk to the left
            elif total_degree_list[i] == -90:
                m = m - 1
            # previous chunk points up -> next chunk up
            elif total_degree_list[i] == 0:
                n = n + 1
            # previous chunk points down -> next chunk down
            elif total_degree_list[i] == -180 or total_degree_list[i] == 180:
                n = n - 1

            position_list.append((m, n))

        return position_list, total_degree_list

    def transform_mn2hv(self, position_list):
        """
        Calculate the parameters from the mn to the hv coordinate system.

        Parameters
        ----------
        position_list : list
            List with positions of all chunks in the mn coordinate system
            including the new chunk.

        Returns
        -------
        size_image_vertikal : int
            vertical size of the whole image
        size_image_horizontal : int
            horizontal size of the whole image
        center_shift_vertikal : int
            vertical shift between mn and hv coordinate system
        center_shift_horizontal : int
            horizontal shift between mn and hv coordinate system
        """
        # Get the number of horizontally arranged chunks
        m_min = min(x[0] for x in position_list)
        m_max = max(x[0] for x in position_list)
        # 1 because of the zero and 4 because of the margin
        m_delta = m_max - m_min + 1 + 4
        size_image_horizontal = int(m_delta * self.size_image_px)

        # Get the number of vertically arranged chunks
        n_min = min(x[1] for x in position_list)
        n_max = max(x[1] for x in position_list)
        n_delta = n_max - n_min + 1 + 4
        size_image_vertikal = int(n_delta * self.size_image_px)

        # Get the translation between the mn and the hv coordinate system
        # 0.5 because the coordinate system is in the center
        # and 2 because of the margin
        m_shift = abs(m_min) + 0.5 + 2
        n_shift = n_max + 0.5 + 2

        # Get the displacement between the coordinate systems h-v and m-n
        center_shift_vertikal = int(n_shift * self.size_image_px)
        center_shift_horizontal = int(m_shift * self.size_image_px)

        return (size_image_vertikal, size_image_horizontal,
                center_shift_vertikal, center_shift_horizontal)

    def select_chunk(self, position_list, total_degree_list):
        """
        Select the next chunk for the road.

        Care is taken to ensure that the snake does not bite its tail.
        The method is simple:
            Check if the cell above is already occupied.
            If not, add the straight line to possible chunks.
            Check if the cell on the right is already occupied.
            If not, add the right curve to possible chunks
            Check if the cell on the left is already occupied.
            If not, add the left curve to possible chunks

        The next chunk is selected randomly.
        The position of the next chunk depends on the position and rotation
        angle of the previous chunk.

        Parameters
        ----------
        position_list : List
            Position of road elements including the position of the new chunk
        total_degree_list : List
            List with resulting angle information of all chunks.
            So the total angle rotation

        Returns
        -------
        None.
        """
        # Remove the first element from the list,
        # through which the car has already passed.
        self.file_list.pop(0)
        self.degree_list.pop(0)

        # Take the position of the current (second to last added) chunk.
        (m, n) = position_list[len(position_list)-1]
        degree = total_degree_list[len(position_list)-1]
        elements = []

        if degree == 0:
            if not (m, n+1) in position_list:
                elements.append("line")
                elements.append("intersection")
            if not (m+1, n) in position_list:
                elements.append("curve_right")
            if not (m-1, n) in position_list:
                elements.append("curve_left")

        elif degree == 90:
            if not (m+1, n) in position_list:
                elements.append("line")
                elements.append("intersection")
            if not (m, n-1) in position_list:
                elements.append("curve_right")
            if not (m, n+1) in position_list:
                elements.append("curve_left")

        elif degree == -90:
            if not (m-1, n) in position_list:
                elements.append("line")
                elements.append("intersection")
            if not (m, n+1) in position_list:
                elements.append("curve_right")
            if not (m, n-1) in position_list:
                elements.append("curve_left")

        elif degree == 180 or degree == -180:
            if not (m, n-1) in position_list:
                elements.append("line")
                elements.append("intersection")
            if not (m-1, n) in position_list:
                elements.append("curve_right")
            if not (m+1, n) in position_list:
                elements.append("curve_left")

        [file] = np.random.choice(elements, 1)
        self.file_list.append(file)

        # Calculate the angle of the new chunk.
        with open(f"chunks/{file}.json", 'r') as openfile:
            json_object = json.load(openfile)
        degree = json_object['degree']
        self.degree_list.append(degree)

    def mn2coords(self, position_list, center_shift_vertikal,
                  center_shift_horizontal):
        """
        Transform coordinates of chunks from the mn- to the hv-system.

        Parameters
        ----------
        position_list : List
            List with the positions of all chunks in the mn coordinate system.
        center_shift_vertikal : int
            vertical offset between mn and hv coordinate system
        center_shift_horizontal : int
            horizontal offset between mn and hv coordinate system

        Returns
        -------
        coords_list : List
            List with the positions of all chunks in the hv coordinate system.
        """
        coords_list = []

        for i in range(0, len(position_list)):
            (m, n) = position_list[i]

            h_1 = center_shift_horizontal + (m - 0.5) * self.size_image_px
            h_2 = center_shift_horizontal + (m + 0.5) * self.size_image_px
            v_1 = center_shift_vertikal - (n + 0.5) * self.size_image_px
            v_2 = center_shift_vertikal - (n - 0.5) * self.size_image_px

            coords_list.append([int(v_1), int(v_2), int(h_1), int(h_2)])

        return coords_list

    def get_drive_points(self, center_shift_vertikal, center_shift_horizontal):
        """
        Get travel points in the coordinate system of the entire image.

        Before the travel points are only known in the coordinate system of
        the individual chunk.


        Parameters
        ----------
        center_shift_vertikal : int
            vertical offset between mn and hv coordinate system
        center_shift_horizontal : int
            horizontal offset between mn and hv coordinate system

        Returns
        -------
        drive_point_coords_list : List
            Travel points of the car in the hv coordinate system
        angles : List
        """
        drive_point_coords_list = []

        json_def = self.chunk_json[self.file_list[0]]
        drive_points = json_def['drive_points']

        for x in drive_points:
            h_coord = x[0] - self.size_image_px/2 + center_shift_horizontal
            v_coord = x[1] - self.size_image_px/2 + center_shift_vertikal
            drive_point_coords_list.append([int(h_coord), int(v_coord)])

        angles = np.linspace(0, json_def["degree"] / 180 * np.pi,
                             len(drive_points))

        return drive_point_coords_list, angles

    def insert_chunk(self, coords_list, total_degree_list, size_image_vertikal,
                     size_image_horizontal, interrupted_lines):
        """
        Insert all chunks into the background image.

        Parameters
        ----------
        coords_list : List
            List with the positions of all chunks in the hv coordinate system
        total_degree_list : List
            List with resulting angle information of all chunks
        size_image_vertikal : int
            Vertical size of background image
        size_image_horizontal : int
            Horizontal size of background image
        interrupted_lines : Bool
            Flag that decides if lane markings are interrupted.

        Returns
        -------
        full_image_nice : Array
            Photorealistic image of the background with inserted chunks.
        full_image_segment : Array
            Annotation of the background with inserted chunks.
        """
        full_image_nice = np.zeros((
            size_image_vertikal, size_image_horizontal), dtype=np.float32)
        full_image_segment = np.zeros((
            size_image_vertikal, size_image_horizontal), dtype=np.float32)

        for i, file in enumerate(self.file_list):
            angle = - total_degree_list[i] if i > 0 else 0
            variant_idx = np.random.randint(0, len(self.images[file]["nice"]))

            img_nice = self.images[file]["nice"][variant_idx][angle]
            img_segment = self.images[file]["segment"][variant_idx][angle]

            [v_1, v_2, h_1, h_2] = coords_list[i]

            full_image_nice[v_1:v_2, h_1:h_2] = img_nice
            full_image_segment[v_1:v_2, h_1:h_2] = img_segment

        return full_image_nice, full_image_segment

    def build_road(self):
        """
        Create annotated background image of the road.

        The following steps are performed for this process:

        1) Determine where the next chunk should be placed.
        2) Determine the parameters to transform the mn coordinate
            system to the hv coordinate system.
        3) Choose a chunk for the next cell.
        4) Transform the position of the chunks from the mn
            to the hv coordinate system.
        5) Determine the travel points in background image coordinates.
        6) Insert all chunks into the background image.

        Parameters
        ----------
        None.

        Returns
        -------
        full_image_nice : Array
            Photorealistic image of the background with inserted chunks.
        full_image_segment : Array
            Annotation of the background with inserted chunks.
        drive_point_coords_list : List
            Travel points of the car in the hv coordinate system
        coords_list : List
            List with the positions of all chunks in the hv coordinate system.
        angles
        """
        position_list, total_degree_list = self.get_position()
        (size_image_vertikal, size_image_horizontal, center_shift_vertikal,
         center_shift_horizontal) = self.transform_mn2hv(position_list)
        self.select_chunk(position_list, total_degree_list)
        coords_list = self.mn2coords(position_list, center_shift_vertikal,
                                     center_shift_horizontal)
        drive_point_coords_list, angles = self.get_drive_points(
            center_shift_vertikal, center_shift_horizontal)
        full_image_nice, full_image_segment = self.insert_chunk(
            coords_list, total_degree_list, size_image_vertikal,
            size_image_horizontal, 0)

        return (full_image_nice, full_image_segment,
                drive_point_coords_list, coords_list, angles)
