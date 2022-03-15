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
    
        
        
    """

    def __init__(self, config):
        self.images = {}
        self.chunk_json = {}
        self.file_list = ["line", "line", "intersection", "line"]
        self.degree_list = [ 0, 0, 0, 0 ]
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
        for segment_type in [ "line", "intersection", "curve_left",
                             "curve_right" ]:
            self.images[segment_type] = { "segment": [], "nice": [] }
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
        The function determines the position of the next chunk.

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
        position_list = [(0,0)]

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

        return size_image_vertikal, size_image_horizontal, center_shift_vertikal, center_shift_horizontal

    def select_chunk(self, position_list, total_degree_list):
        """
        Schritt 3
        Diese Funktion wählt ein Straßenelement für den nächsten chunk aus. Dabei wird darauf geachtet, dass sich die Schlange nicht in den Schwanz beißt
        Input: file_list(Namen aller Straßenelemente der bisherigen Straße) ist ein Klassenattribut, 
        position_list(Position aller Straßenelemente inklusive der Position des neuen chunks),
        degree_list(Liste mit Winkelinformationen zu jedem bisherigen chunk) ist ein Klassenattribut, 
        total_degree_list(Liste mit resultierender Winkelinformation aller chunks. Also die Gesamtwinkeldrehung)
        Output: file_list(Namen aller Straßenelemente der bisherigen Straße inklusive des Namens des neuen chunks), 
        degree_list(Liste mit Winkelinformationen zu jedem bisherigen chunk inklusive des Winkels des neuen chunks)
        Die Klassenattribute werden verändert. Es gibt keine direkte Funktionsausgabe
        """
        # Entferne erstes Element aus list, durch welches das Auto schon durchgefahren ist
        self.file_list.pop(0)
        self.degree_list.pop(0)

        # Nehme die Position des aktuellen (vorletzten hinzugefügten) chunks. Nicht den neusten chunk
        (m, n) = position_list[len(position_list)-1]
        degree = total_degree_list[len(position_list)-1]
        elements = []

        # Wenn das Ende des aktuelle chunks in positive n-Achsenrichtung zeigt, prüfe Folgendes:
        if degree == 0:
            # Prüfe ob das Element relativ oben schon belegt ist. Falls nein, füge line hinzu
            if not (m, n+1) in position_list:
                elements.append("line")
                elements.append("intersection")
            # Prüfe ob das Element relativ rechts schon belegt ist. Falls nein, füge curve_right hinzu
            if not (m+1, n) in position_list:
                elements.append("curve_right")
            # Prüfe ob das Element relativ links schon belegt ist. Falls nein, füge curve_left hinzu
            if not (m-1, n) in position_list:
                elements.append("curve_left")

        # Wenn das Ende des aktuelle chunks in positive m-Achsenrichtung zeigt, prüfe Folgendes:
        elif degree == 90:
            # Prüfe ob das Element relativ oben schon belegt ist. Falls nein, füge line hinzu
            if not (m+1, n) in position_list:
                elements.append("line")
                elements.append("intersection")
            # Prüfe ob das Element relativ rechts schon belegt ist. Falls nein, füge curve_right hinzu
            if not (m, n-1) in position_list:
                elements.append("curve_right")
            # Prüfe ob das Element relativ links schon belegt ist. Falls nein, füge curve_left hinzu
            if not (m, n+1) in position_list:
                elements.append("curve_left")

        # Wenn das Ende des aktuelle chunks in negative m-Achsenrichtung zeigt, prüfe Folgendes:
        elif degree == -90:
            # Prüfe ob das Element relativ oben schon belegt ist. Falls nein, füge line hinzu
            if not (m-1, n) in position_list:
                elements.append("line")
                elements.append("intersection")
            # Prüfe ob das Element relativ rechts schon belegt ist. Falls nein, füge curve_right hinzu
            if not (m, n+1) in position_list:
                elements.append("curve_right")
            # Prüfe ob das Element relativ links schon belegt ist. Falls nein, füge curve_left hinzu
            if not (m, n-1) in position_list:
                elements.append("curve_left")

        # Wenn das Ende des aktuelle chunks in negative n-Achsenrichtung zeigt, prüfe Folgendes:
        elif degree == 180 or degree == -180:
            # Prüfe ob das Element relativ oben schon belegt ist. Falls nein, füge line hinzu
            if not (m, n-1) in position_list:
                elements.append("line")
                elements.append("intersection")
            # Prüfe ob das Element relativ rechts schon belegt ist. Falls nein, füge curve_right hinzu
            if not (m-1, n) in position_list:
                elements.append("curve_right")
            # Prüfe ob das Element relativ links schon belegt ist. Falls nein, füge curve_left hinzu
            if not (m+1, n) in position_list:
                elements.append("curve_left")
                
        # Zum debuggen
        #elements = ["line", "intersection"]
        # Zum debuggen

        # Wähle ein zufälliges Element aus den erlaubten Streckenelementen
        [file] = np.random.choice(elements, 1)  #, p=probability
        self.file_list.append(file)

        # Wähle ein zufälliges gestörtes Bild

        # Bestimme den Winkel des neuen chunks (erst für die nächste Iteration von Bedeutung) und hefte es hinten an.
        with open(f"chunks/{file}.json", 'r') as openfile: 
            json_object = json.load(openfile)
        degree = json_object['degree']
        self.degree_list.append(degree)

    def mn2coords(self, position_list, center_shift_vertikal, center_shift_horizontal):
        """
        Schritt 4
        Transformiere die Position der chunks vom mn- in das hv-Koordinatensystem
        Input: position_list(Liste mit den Positionen aller chunks im mn-Koordinatensystem), 
        size_image_px(Größe eines chunks in Pixel) ist ein Klassenattribut, 
        center_shift_vertikal(vertikaler Abstand zwischen mn- und hv-Koordinatensystem), 
        center_shift_horizontal(horizontaler Abstand zwischen mn- und hv-Koordinatensystem)
        Output: coords_list(Liste mit den Positionen aller chunks im hv-Koordinatensystem)
        """
        coords_list = []

        # Iteriere über alle chunks und transformiere die Punkte in das hv-Koordinatensystem
        for i in range(0, len(position_list)):
            (m,n) = position_list[i]

            h_1 = center_shift_horizontal + (m - 0.5) * self.size_image_px
            h_2 = center_shift_horizontal + (m + 0.5) * self.size_image_px
            v_1 = center_shift_vertikal - (n + 0.5) * self.size_image_px
            v_2 = center_shift_vertikal - (n - 0.5) * self.size_image_px

            coords_list.append([int(v_1), int(v_2), int(h_1), int(h_2)])

        return coords_list

    def get_drive_points(self, center_shift_vertikal, center_shift_horizontal):
        """
        Schritt 5
        Diese Funktion berechnet die Fahrtpunkte des ersten chunks aus der chunk-Kette in Gesamtbildkoordinaten um
        Input: center_shift_vertikal(vertikaler Abstand zwischen mn- und hv-Koordinatensystem), 
        center_shift_horizontal(horizontaler Abstand zwischen mn- und hv-Koordinatensystem), 
        size_image_px(Größe eines chunks in Pixel), file_list(Namen aller Straßenelemente der bisherigen Straße inklusive des Namens des neuen chunks)
        Die letzten beiden Argumente sind Klassenattribute.
        Output: drive_point_coords_list(Fahrtpunkte des Autos im hv-Koordinatensystem)
        """
        drive_point_coords_list = []

        file = self.file_list[0]
        # Bestimme den Winkel des neuen chunks (erst für die nächste Iteration von Bedeutung) und hefte es hinten an.
        json_def = self.chunk_json[self.file_list[0]]
        drive_points = json_def['drive_points']

        for x in drive_points:
            h_coord = x[0] - self.size_image_px/2 + center_shift_horizontal
            v_coord = x[1] - self.size_image_px/2 + center_shift_vertikal
            drive_point_coords_list.append([int(h_coord), int(v_coord)])

        angles = np.linspace(0, json_def["degree"] / 180  * np.pi, len(drive_points))

        return drive_point_coords_list, angles

    def insert_chunk(self, coords_list, total_degree_list, size_image_vertikal, size_image_horizontal, interrupted_lines):
        """
        Schritt 6
        Füge alle chunks in ein Gesamtbild ein
        Input: file_list(Namen aller Straßenelemente der bisherigen Straße inklusive des Namens des neuen chunks) ist ein Klassenattribut, 
        coords_list(Liste mit den Positionen aller chunks im hv-Koordinatensystem),
        total_degree_list(Liste mit resultierender Winkelinformation aller chunks), 
        size_image_vertikal(vertikale Größe des Gesamtbildes), size_image_horizontal(horizontale Größe des Gesamtbildes), 
        size_image_px(Größe eines chunks in Pixel) ist ein Klassenattribut
        interrupted_lines(Boolsche Variable, die festlegt, ob die Fahrbahnmarkierungen Lücken haben)
        Output: full_image_nice(Gesamtbild der Straße - der Realität nachgeahmt), full_image_segment(Gesamtbild der Straße - segmentiert)
        """
        # Generiere neues Vollbild
        full_image_nice  = np.zeros((size_image_vertikal, size_image_horizontal), dtype=np.float32)
        full_image_segment = np.zeros((size_image_vertikal, size_image_horizontal), dtype=np.float32)

        # Iteriere über alle chunks
        for i, file in enumerate(self.file_list):
            # Importiere die chunk-Bilder und das JSON-file
            # Lese des Gesamtwinkel aller vorherigen chunks
            angle = - total_degree_list[i] if i > 0 else 0
            variant_idx = np.random.randint(0, len(self.images[file]["nice"]))

            img_nice = self.images[file]["nice"][variant_idx][angle]
            img_segment = self.images[file]["segment"][variant_idx][angle]        

            # Jeder chunk wird entsprechend dem Winkel und der Position seiner Vorgänger platziert
            [v_1, v_2, h_1, h_2] = coords_list[i]

            # Setze chunk-Bild in Vollbild ein
            full_image_nice[v_1:v_2, h_1:h_2] = img_nice
            full_image_segment[v_1:v_2, h_1:h_2] = img_segment

        return full_image_nice, full_image_segment

    def road_debugging(self, full_image_nice, full_image_segment):     
        """
        Diese Funktion stellt die Hintergrundbilder und ein paar nützliche Infos dar.
        Sie ist nur zu Debugging erstellt - eine Bug-Hunter-Funktion.
        """
        full_image_nice = cv.resize(full_image_nice, (500,500), interpolation = cv.INTER_AREA)
        full_image_segment = cv.resize(full_image_segment, (500,500), interpolation = cv.INTER_AREA)

        cv.imshow("nice", full_image_nice) 
        cv.imshow("segment", full_image_segment)    
        
        cv.waitKey(0)   
        cv.destroyAllWindows()  

    def build_road(self):
        """
        Diese Funktion führt alle Funktionen dieser Klasse in dieser Abfolge aus.
        1) Bestimme an welche Stelle der nächste chunk gesetzt wird
        2) Bestimme die Parameter um das mn- in das hv- Koordinatensystem zu transformieren
        3) Wähle ein Straßenelement für den nächsten chunk aus
        4) Transformiere die Position der chunks vom mn- in das hv-Koordinatensystem
        5) Bestimme die Fahrtpunkte in Gesamtbildkoordinaten
        6) Füge alle chunks in ein Gesamtbild ein
        Input: ein road-Objekt
        Output: Zwei Gesamtbilder der Straße (segmentiert und schön), die Koordinaten der Fahrttrajektorie
        """
        
        position_list, total_degree_list = self.get_position()
        size_image_vertikal, size_image_horizontal, center_shift_vertikal, center_shift_horizontal = self.transform_mn2hv(position_list)
        self.select_chunk(position_list, total_degree_list)
        coords_list = self.mn2coords(position_list, center_shift_vertikal, center_shift_horizontal)
        drive_point_coords_list, angles = self.get_drive_points(center_shift_vertikal, center_shift_horizontal)     
        full_image_nice, full_image_segment = self.insert_chunk(coords_list, total_degree_list, size_image_vertikal, size_image_horizontal, 0)

        return full_image_nice, full_image_segment, drive_point_coords_list, coords_list, angles




