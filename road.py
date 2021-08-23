from imports import *

class Road:
    """
    Die Klasse enthält alle Funktionen, die ein Gesamtbild der Straße erzeugen.

    """

    def __init__(self):

        self.file_list = Config.file_list        
        self.degree_list = Config.degree_list
        self.disorder_list = Config.disorder_list
        self.size_image_px = Config.size_image_px 
        
        
    def get_position(self):
        """
        Schritt 1
        Die Funktion bestimmt die Position des nächsten chunks in Abhängigkeit der bisherigen chunk-Winkel im mn-Koordinatensystem.
        Input: degree_list(Liste mit Winkelinformationen zu jedem bisherigen chunk) ist ein Klassenattribut.
        Output: position_list(Liste mit Positionen aller chunks im mn-Koordinatensystem), 
        total_degree_list(Liste mit resultierender Winkelinformation aller chunks)
        """ 
        # Setze den ersten chunk in den Ursprung des mn-Koordinatensystems
        position_list = [(0,0)]

        # Das erste Element beginnt immer orthogonal zum Bild gesehen 
        total_degree_list = [0] 

        # Setze den zweiten bis letzten chunk
        for i in range(1, len(self.degree_list)):

            # Summiere alle bisherigen Winkel um gesamte Drehung zu ermitteln
            total_degree = sum(self.degree_list[1:i+1])
            # print(total_degree)

            # Schreibe die Gesamtdrehung in des entsprechende list
            total_degree_list.append(total_degree)

            # Hole die Position des vorherigen chunks
            (m, n) = position_list[i-1]

            # vorheriger chunk zeigt nach rechts -> nächster chunk rechts
            if total_degree_list[i] == 90:
                m = m + 1
            # vorheriger chunk zeigt nach links -> nächster chunk links
            elif total_degree_list[i] == -90:
                m = m - 1
            # vorheriger chunk zeigt nach oben -> nächster chunk oben
            elif total_degree_list[i] == 0:
                n = n + 1
            # vorheriger chunk zeigt nach unten -> nächster chunk unten
            elif total_degree_list[i] == -180 or total_degree_list[i] == 180:
                n = n - 1

            # Schreibe die Position des aktuellen chunks
            position_list.append((m, n))
        
        return position_list, total_degree_list


    def transform_mn2hv(self, position_list):
        """
        Schritt 2
        Diese Funktion berechnet die Transformationsparameter vom mn- in das hv-Koordinatensystem
        Input: position_list(Liste mit Positionen aller chunks im mn-Koordinatensystem inklusive des neuen chunks), 
        size_image_px(Größe eines chunks in Pixel) ist ein Klassenattribut.
        Output: size_image_vertikal(vertikale Größe des Gesamtbildes), size_image_horizontal(horizontale Größe des Gesamtbildes), 
        center_shift_vertikal(vertikaler Abstand zwischen mn- und hv-Koordinatensystem), 
        center_shift_horizontal(horizontaler Abstand zwischen mn- und hv-Koordinatensystem)
        """
        # Bestimme die Anzahl horizontal angeordneter chunks
        m_min = min(x[0] for x in position_list)
        m_max = max(x[0] for x in position_list)
        m_delta = m_max - m_min + 1 + 4              # 1 wegen der Null und 4 wegen dem Rand
        size_image_horizontal = int(m_delta * self.size_image_px)

        # Bestimme die Anzahl vertikal angeordneter chunks
        n_min = min(x[1] for x in position_list)
        n_max = max(x[1] for x in position_list)
        n_delta = n_max - n_min + 1 + 4		# 1 wegen der Null und 4 wegen dem Rand
        size_image_vertikal = int(n_delta * self.size_image_px)

        # Bestimme die Translation zwischen dem mn- und dem hv-Koordinatensystem
        m_shift = abs(m_min) + 0.5 + 2              # 0.5 weil das Koordinatensystem in der Mitte steht und 2 wegen dem Rand
        n_shift = n_max + 0.5 + 2

        # Bestimme die Verschiebung zwischen den Koordinatensystemen h-v und m-n
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
        self.disorder_list.pop(0)

        # Nehme die Position des aktuellen (vorletzten hinzugefügten) chunks. Nicht den neusten chunk
        (m, n) = position_list[len(position_list)-1]
        degree = total_degree_list[len(position_list)-1]
        elements = []
        probability = []

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

        #elements = ["line", "intersection"]  # zum Debuggen

        # Wähle ein zufälliges Element aus den erlaubten Streckenelementen
        [file] = np.random.choice(elements, 1)  #, p=probability
        self.file_list.append(file)

        # Wähle ein zufälliges gestörtes Bild
        self.disorder_list.append(random.randint(0,9))

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
        with open(f"chunks/{file}.json", 'r') as openfile: 
            json_object = json.load(openfile) 
        drive_points = json_object['drive_points']

        for x in drive_points:
            h_coord = x[1] - self.size_image_px/2 + center_shift_horizontal
            v_coord = x[0] - self.size_image_px/2 + center_shift_vertikal
            drive_point_coords_list.append([int(h_coord), int(v_coord)])

        return drive_point_coords_list


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
        full_image_nice  = np.zeros((size_image_vertikal, size_image_horizontal))
        full_image_segment = np.zeros((size_image_vertikal, size_image_horizontal, 4))


        # Iteriere über alle chunks
        for i in range(0, len(self.file_list)):

            # Lese aktuellen chunk Namen
            file = self.file_list[i]

            # Soll die Fahrbahnmarkierung lückenhaft sein?
            if True:
                # Berücksichtige bei der Wahl gestörte Bilder
                number = self.disorder_list[i]

            # Importiere die chunk-Bilder und das JSON-file
            img_nice = cv.imread(f"chunks/{file}_nice_{number}.jpg", cv.IMREAD_GRAYSCALE)                        
            img_segment = cv.imread(f"chunks/{file}_segment.png", cv.IMREAD_UNCHANGED)        
        
            # Lese des Gesamtwinkel aller vorherigen chunks
            angle = - total_degree_list[i] if i > 0 else 0

            # Drehe die chunk-Bilder um diesen Winkel
            rot_mat = cv.getRotationMatrix2D((self.size_image_px/2, self.size_image_px/2), angle, 1.0)
            img_nice = cv.warpAffine(img_nice, rot_mat, (self.size_image_px, self.size_image_px))
            img_segment = cv.warpAffine(img_segment, rot_mat, (self.size_image_px, self.size_image_px))

            # Jeder chunk wird entsprechend dem Winkel und der Position seiner Vorgänger platziert
            [v_1, v_2, h_1, h_2] = coords_list[i]

            for i in range(0,3):
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
        
        position_list, total_degree_list = Road.get_position(self)
        size_image_vertikal, size_image_horizontal, center_shift_vertikal, center_shift_horizontal = Road.transform_mn2hv(self, position_list)
        Road.select_chunk(self, position_list, total_degree_list)
        coords_list = Road.mn2coords(self, position_list, center_shift_vertikal, center_shift_horizontal)
        drive_point_coords_list = Road.get_drive_points(self, center_shift_vertikal, center_shift_horizontal)     
        interruption = cv.getTrackbarPos("interruption", "control")
        full_image_nice, full_image_segment = Road.insert_chunk(self, coords_list, total_degree_list, size_image_vertikal, size_image_horizontal, interruption)

        return full_image_nice, full_image_segment, drive_point_coords_list, coords_list



