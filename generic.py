from imports import *

"""
Die Klasse 'Generic' ist ein Template für alle Streckenelemente.
Sie enthält die Parametrisierung der Fahrbahn und Funktionsdeklarationen.
"""


class Generic:


    def __init__(self, name, path):
        
        # Siehe Bedeutung in config.py
        self.size_image_px = Config.size_image_px
        self.line_nice_width_px = Config.line_nice_width_px
        self.center_distance_px = Config.center_distance_px
        self.line_segmentated_width_px = Config.line_segmentated_width_px
        self.street_width_px = Config.street_width_px
        self.drive_period_px = Config.drive_period_px
        self.dot_period_px = Config.dot_period_px
        self.dot_length_px = Config.dot_length_px

        # Name des chunk-Objekts
        self.name = name

        # Wo sollen die Dateien gespeichert werden
        self.path = path



    def draw_lines(self):
        # Die Funktionsdefinition folgt in den Kindklassen
        return None

    def create_metadata(self):
        # Die Funktionsdefinition folgt in den Kindklassen
        return None

    def debug_chunk():
        # Die Funktionsdefinition folgt in den Kindklassen
        return None

    def create_chunk(self):
        # Die Funktionsdefinition folgt in den Kindklasse
        return None