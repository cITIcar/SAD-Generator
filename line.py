from imports import *

class Line(Generic):
    """
    Die Klasse erbt von der Elternklasse 'Generic'. 
    Ihre Funktionen erzeugen ein realistischen Straßenbild, 
    ein segmentiertes Straßenbild und eine JSON-Datei mit den Meta-Daten.
    """

    def __init__(self, name, path):
        Generic.__init__(self, name, path)


    def draw_lines(self):
        """Diese Funktion generiert ein Streckenelement samt Etikette. 
        Dieses wird später nach dem Carcasonne-Prinzip aneinandergesetzt.
        Output: 
        img_nice (Streckenelement als schönes Bild), img_segment (Etikette des Streckenelements)
        """       
        # Generierung des schwarzen Hintergrunds
        img_nice = np.zeros((self.size_image_px,self.size_image_px))
        img_segment = np.zeros((self.size_image_px,self.size_image_px, 4))
        # img_segment[:,:,0] = np.ones((self.size_image_px,self.size_image_px))*50        
        
        # Definiere linke Markierung im schönen Bild
        start_nice_left = self.size_image_px/2 - self.street_width_px - self.line_nice_width_px/2
        end_nice_left = self.size_image_px/2 - self.street_width_px + self.line_nice_width_px/2

        # Definiere mittlere Markierung im schönen Bild 
        start_nice_center = self.size_image_px/2 - self.line_nice_width_px/2
        end_nice_center = self.size_image_px/2 + self.line_nice_width_px/2

        # Definiere rechte Markierung im schönen Bild
        start_nice_right = self.size_image_px/2 + self.street_width_px - self.line_nice_width_px/2
        end_nice_right = self.size_image_px/2 + self.street_width_px + self.line_nice_width_px/2

        # Definiere linke Markierung im segmentierten Bild
        start_seg_left = self.size_image_px/2 - self.street_width_px - self.line_segmentated_width_px/2
        end_seg_left = self.size_image_px/2 - self.street_width_px + self.line_segmentated_width_px/2

        # Definiere mittlere Markierung im segmentierten Bild
        start_seg_center = self.size_image_px/2 - self.line_segmentated_width_px/2
        end_seg_center = self.size_image_px/2 + self.line_segmentated_width_px/2

        # Definiere rechte Markierung im segmentierten Bild
        start_seg_right = self.size_image_px/2 + self.street_width_px - self.line_segmentated_width_px/2
        end_seg_right = self.size_image_px/2 + self.street_width_px + self.line_segmentated_width_px/2

        # Zeichne Randlinie links im schönen Bild
        img_nice[:, int(start_nice_left) : int(end_nice_left)] = 255

        # Zeichne Randlinie rechts im schönen Bild
        img_nice[:, int(start_nice_right) : int(end_nice_right)] = 255

        # Zeichne Mittellinie im schönen Bild
        for n in range (0,int(self.size_image_px/self.dot_period_px) + 1):
            img_nice[n * self.dot_period_px : n * self.dot_period_px + self.dot_length_px , int(start_nice_center) : int(end_nice_center)] = 255

        # Markiere die linke Fahrbahn im segmentierten Bild
        img_segment[:, int(start_seg_left) : int(self.size_image_px/2), 1] = 200

        # Markiere die rechte Fahrbahn im segmentierten Bild
        img_segment[:, int(self.size_image_px/2): int(end_seg_right), 2] = 200

        # Speicherung der Bilder
        cv.imwrite(f'{self.path}/line_nice_0.jpg', img_nice)
        cv.imwrite(f'{self.path}/line_segment.png', img_segment)

        # Umwandlung in eine JSON kompatible Liste
        #img_segment = img_segment.tolist()

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

        # Bestimme Fahrtpunkte im chunkfesten Koordinatensystem
        for n in range (0, int(self.size_image_px / self.drive_period_px) + 1):
            x_value = self.size_image_px - n * self.drive_period_px
            y_value = self.size_image_px / 2 + self.center_distance_px
            drive_points.append((int(x_value), int(y_value)))


        # Erstellung einer json-Datei
        dictionary = { 
              "name" : self.name, 
              "degree" : 0,
              "drive_points" : drive_points
              } 

        # Speicherung der json-Datei
        with open(f'{self.path}/line.json', "w") as outfile:  
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
        
        img_segment = np.argmax(img_segment, axis = -1).astype(float) # CV erwartet den Datentyp float             
        img_segment = cv.resize(img_segment, (500, 500), interpolation = cv.INTER_AREA)
        
        
        # Bilder darstellen
        cv.imshow("nice",img_nice)
        cv.imshow("segment",img_segment/2)
        cv.waitKey(0)  
        cv.destroyAllWindows() 
        

    def create_chunk(self):
        # Diese Funktion fasst die oben genannten Funktionen zusammen.
        img_nice, img_segment = Line.draw_lines(self)
        drive_points = Line.create_metadata(self)
        Line.debug_chunk(img_nice, img_segment, drive_points)
