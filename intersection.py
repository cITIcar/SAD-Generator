from imports import *

class Intersection(Generic):
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
        img_nice[0 : int(self.size_image_px/2 - self.street_width_px), int(start_nice_left) : int(end_nice_left)] = 255
        img_nice[int(self.size_image_px/2 + self.street_width_px) : self.size_image_px, int(start_nice_left) : int(end_nice_left)] = 255

        # Zeichne Randlinie rechts im schönen Bild
        img_nice[0 : int(self.size_image_px/2 - self.street_width_px), int(start_nice_right) : int(end_nice_right)] = 255
        img_nice[int(self.size_image_px/2 + self.street_width_px) : self.size_image_px, int(start_nice_right) : int(end_nice_right)] = 255
        
        # Zeichne Querlinien im schönen Bild
        # oben links
        x = int(self.size_image_px/2 - self.street_width_px)
        height = self.line_nice_width_px
        y = 0
        width = int(end_nice_left)
        img_nice[x : x+height, y : y+width] = 255
        
        #oben rechts
        x = int(self.size_image_px/2 - self.street_width_px)
        height = self.line_nice_width_px
        y = int(start_nice_right)
        width = self.size_image_px-y
        img_nice[x : x+height, y : y+width] = 255

        # unten links
        x = int(self.size_image_px/2 + self.street_width_px)
        height = self.line_nice_width_px
        y = 0
        width = int(end_nice_left)
        img_nice[x : x+height, y : y+width] = 255

        # unten rechts
        x = int(self.size_image_px/2 + self.street_width_px)
        height = self.line_nice_width_px
        y = int(start_nice_right)
        width = self.size_image_px-y
        img_nice[x : x+height, y : y+width] = 255

        # Zeichne Haltelinien
        x = int(self.size_image_px/2 - self.street_width_px - self.line_nice_width_px )
        height = self.line_nice_width_px * 2
        y = int(end_nice_left)
        width = int(self.size_image_px/2 - end_nice_left + self.line_nice_width_px / 2)
        img_nice[x : x+height, y : y+width] = 255

        x = int(self.size_image_px/2 + self.street_width_px)
        height = self.line_nice_width_px * 2
        y = int(start_nice_right)
        y = int(self.size_image_px/2 - self.line_nice_width_px / 2)
        width = int(start_nice_right - self.size_image_px/2 + self.line_nice_width_px)
        img_nice[x : x+height, y : y+width] = 255

        # Zeichne Mittellinie im schönen Bild
        for n in range (0,int((self.size_image_px-self.street_width_px*2)/self.dot_period_px/2)):
            img_nice[n * self.dot_period_px : n * self.dot_period_px + self.dot_length_px , int(start_nice_center) : int(end_nice_center)] = 255
            img_nice[int(self.size_image_px / 2 + self.street_width_px + n * self.dot_period_px) : int(self.size_image_px / 2 + self.street_width_px + n * self.dot_period_px + self.dot_length_px) , int(start_nice_center) : int(end_nice_center)] = 255
        
        for n in range (0,int((self.size_image_px-self.street_width_px*2)/self.dot_period_px/2)):
            img_nice[int(start_nice_center) : int(end_nice_center), n * self.dot_period_px : n * self.dot_period_px + self.dot_length_px ] = 255
            img_nice[int(start_nice_center) : int(end_nice_center), int(self.size_image_px / 2 + self.street_width_px + n * self.dot_period_px) : int(self.size_image_px / 2 + self.street_width_px + n * self.dot_period_px + self.dot_length_px)] = 255


        #img_nice[0 : self.dot_length_px , int(start_nice_center) : int(end_nice_center)] = 255
        #img_nice[2 * self.dot_period_px : 2 * self.dot_period_px + self.dot_length_px , int(start_nice_center) : int(end_nice_center)] = 255

        """
        Hier wird der Code für die Kreuzungsettikette geschrieben
        
        """
        """
        # Markiere die linke Fahrbahn im segmentierten Bild
        img_segment[int(start_seg_left):int(self.size_image_px/2), 0 : int(self.size_image_px/2 - self.street_width_px),2] = 200
        img_segment[int(start_seg_left):int(self.size_image_px/2), int(self.size_image_px/2 + self.street_width_px) : self.size_image_px, 1] = 200
        
        # Markiere die rechte Fahrbahn im segmentierten Bild
        img_segment[int(self.size_image_px/2): int(end_seg_right), 0 : int(self.size_image_px/2 - self.street_width_px), 1] = 200
        img_segment[int(self.size_image_px/2): int(end_seg_right), int(self.size_image_px/2 + self.street_width_px) : self.size_image_px, 2] = 200        
        """

        # Markiere die linke Fahrbahn im segmentierten Bild
        img_segment[ : , int(start_seg_left):int(self.size_image_px/2), 1] = 200 
        
        # Markiere die rechte Fahrbahn im segmentierten Bild
        img_segment[0 : int(self.size_image_px/2 + 0.5 * self.street_width_px), int(self.size_image_px/2): int(end_seg_right), 2] = 200
        img_segment[int(self.size_image_px/2 + self.street_width_px) : self.size_image_px, int(self.size_image_px/2): int(end_seg_right), 2] = 200    

        # Markiere die Kreuzung im segmentierten Bild
        img_segment[int(self.size_image_px/2 + 0.5 * self.street_width_px):int(self.size_image_px/2 + self.street_width_px), int(self.size_image_px/2):int(end_seg_right), 3] = 200
        
        # Speicherung der Bilder
        cv.imwrite(f'{self.path}/intersection_nice_0.jpg', img_nice)
        cv.imwrite(f'{self.path}/intersection_segment.png', img_segment)

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
        with open(f'{self.path}/intersection.json', "w") as outfile: 
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
        cv.imshow("segment",img_segment/4)
        cv.waitKey(0)  
        cv.destroyAllWindows() 



    def create_chunk(self):
        # Diese Funktion fasst die oben genannten Funktionen zusammen.
        img_nice, img_segment = Intersection.draw_lines(self)
        drive_points = Intersection.create_metadata(self)
        Intersection.debug_chunk(img_nice, img_segment, drive_points)

