from imports import *


class Curve(Generic):
    """
    Die Klasse erbt von der Elternklasse 'Generic'. 
    Ihre Funktionen erzeugen ein realistischen Straßenbild, 
    ein segmentiertes Straßenbild und eine JSON-Datei mit den Meta-Daten.
    """

    def __init__(self, name, path, direction):
        Generic.__init__(self, name, path)
        # Links oder Rechts "right" oder "left" sind gültig
        self.direction = direction


    def draw_lines(self):
        """
        Diese Funktion generiert ein Streckenelement samt Etikette. 
        Dieses wird später nach dem Carcasonne-Prinzip aneinandergesetzt.
        Output: 
        img_nice (Streckenelement als schönes Bild), img_segment (Etikette des Streckenelements)
        """
        
        # Generierung des schwarzen Hintergrunds
        img_nice = np.zeros((self.size_image_px,self.size_image_px))
        img_segment = np.zeros((self.size_image_px,self.size_image_px, 4))
        # img_segment[:,:,0] = np.ones((self.size_image_px,self.size_image_px))*50        
    
        # Bestimme den Mittelpunkt des Kreises
        center = (self.size_image_px, self.size_image_px) if self.direction == "right" else (0, self.size_image_px)
        
        # Bestimme die Radien der Markierung
        short_radius = int(self.size_image_px / 2) - self.street_width_px
        medium_radius = int(self.size_image_px / 2)
        large_radius = int(self.size_image_px / 2) + self.street_width_px

        right_lane_radius = short_radius if self.direction == "right" else large_radius
        left_lane_radius = large_radius if self.direction == "right" else short_radius

        start_degree = 180 if self.direction == "right" else 270
        end_degree = 270 if self.direction == "right" else 360

        color = (255,255,255)

        # Definiere linke Markierung im schönen Bild
        cv.ellipse(img_nice, center, (left_lane_radius, left_lane_radius), 0, start_degree, end_degree, color, self.line_nice_width_px)

        # Definiere rechte Markierung im schönen Bild 
        cv.ellipse(img_nice, center, (right_lane_radius, right_lane_radius), 0, start_degree, end_degree, color, self.line_nice_width_px)


        # gekrümmte Periode [°]
        dot_period_degree = int(self.dot_period_px * 180 / (self.size_image_px/2 * math.pi))
        # gekrümmte Pulsweite  [°]
        dot_length_degree = int(self.dot_length_px * 180 / (self.size_image_px/2 * math.pi))
        # Mittellinie
        for n in range (0,int(90 / dot_period_degree)+1):
          cv.ellipse(img_nice, center, (medium_radius, medium_radius), 0, start_degree + n * dot_period_degree, start_degree + n * dot_period_degree + dot_length_degree, color, self.line_nice_width_px)


        left_middle_radius = int((left_lane_radius + medium_radius)/2)
        right_middle_radius = int((right_lane_radius + medium_radius)/2)
        segment_color = (1, 1, 1)
        
        self.line_segmentated_width_px = int(abs(left_lane_radius - right_lane_radius)/2)

        # Markiere die linke Fahrbahn im segmentierten Bild
        cv.ellipse(img_segment, center, (left_middle_radius, left_middle_radius), 0, start_degree, end_degree, (0,200,0), self.line_segmentated_width_px)

        # Markiere die rechte Fahrbahn im segmentierten Bild
        cv.ellipse(img_segment, center, (right_middle_radius, right_middle_radius), 0, start_degree, end_degree, (0,0,200), self.line_segmentated_width_px)

        # Speicherung der Bilder
        cv.imwrite(f'{self.path}/curve_{self.direction}_nice_0.jpg', img_nice)
        cv.imwrite(f'{self.path}/curve_{self.direction}_segment.png', img_segment)

        return img_nice, img_segment


    def create_metadata(self):
        """
        Diese Funktion speicher das Streckenelement samt Etikette an dem vorgegebenen Pfad.
        Außerdem wird eine JSON-Datei erzeugt, die Metadaten des Streckenelements enthält.
        Zudem werden die Fahrpunkte des Fahrzeugs innerhalb des Streckenelements bestimmt.
        Output:
        drive_points (Punkte auf denen das Auto fährt)
        """
        drive_points = []
        box_points = []

        # gekrümmte Fahrtpunktabstand [°]
        drive_point_rad = int(self.drive_period_px * 180 / ((self.size_image_px/2 + self.center_distance_px) * math.pi))*math.pi/180
        counter = int(math.pi/(2*drive_point_rad))+1

        # Fahrtpunkte
        if self.direction == "right":

          for n in reversed(range(0, counter)):
            x_value = self.size_image_px + (-self.center_distance_px + self.size_image_px/2)*math.cos(math.pi + n*drive_point_rad)
            y_value = self.size_image_px + (-self.center_distance_px + self.size_image_px/2)*math.sin(math.pi +n*drive_point_rad)
            
            drive_points.append((int(x_value), int(y_value)))


        elif self.direction == "left":

          for n in range (0, counter):
            x_value = self.size_image_px + (self.center_distance_px + self.size_image_px/2)*math.cos(math.pi * 0.5 + n*drive_point_rad)
            y_value = 0 + (self.center_distance_px + self.size_image_px/2)* math.sin(math.pi * 0.5 +n*drive_point_rad)
           
            drive_points.append((int(x_value), int(y_value)))


        # Erstellung der json-Datei
        dictionary = { 
            "name" : "curve_"+self.direction, 
            "degree" : 90 if self.direction == "right" else -90,
            "drive_points" : drive_points
        } 

        with open(f'{self.path}/curve_{self.direction}.json', "w") as outfile: 
            json.dump(dictionary, outfile) 


        return drive_points


    def debug_chunk(img_nice, img_segment, drive_points):
        """
        Diese Funktion visualisiert das Streckenelement samt Etikette in einem Fenster.
        Die Fahrpunkte werden innerhalb der Bildes markiert.
        """        
        # Fahrtpunkte in Bild anzeichen
        for point in drive_points:
            cv.circle(img_nice,(point[1], point[0]), 1, (255,255,255), -1)

        
        img_nice = cv.resize(img_nice, (500, 500), interpolation = cv.INTER_AREA)
        
        img_segment = np.argmax(img_segment, axis = -1).astype(float)
        img_segment = cv.resize(img_segment, (500, 500), interpolation = cv.INTER_AREA)

        # Bilder darstellen
        cv.imshow("nice",img_nice)
        cv.imshow("segment",img_segment/2)
        cv.waitKey(0)  
        cv.destroyAllWindows() 


    def create_chunk(self):
        # Diese Funktion fasst die oben genannten Funktionen zusammen.
        img_nice, img_segment = Curve.draw_lines(self)
        drive_points = Curve.create_metadata(self)
        Curve.debug_chunk(img_nice, img_segment, drive_points)





