from imports import *
from numpy import random

class Visualization:
    
    """
    Die Klasse enthält Funktionen, die bei jedem Schritt des Autos auf der Trajektorie durchgeführt werden müssen.
    Die Funktionen generieren aus dem Carcassone-Spielbrett die eigentliche Carolo-Simulation
    """

    def __init__(self):

        self.h_size_px = Config.h_size_px      
        self.v_size_px = Config.v_size_px
        self.index = 1
        self.i = 0
        self.full_image_nice = None
        self.full_image_segment = None
        self.drive_point_coords_list = None
        self.car_angle_list = None
        self.abcd_points_k2 = None
        self.drunk = True
                                
                                
    def get_ABCD(self):
        """
        Schritt 1 
        Diese Funktion berechnet die Punkte A,B,C,D in Autokoordinaten.
        Es handelt sich um die Eckpunkte des ROIs.
        Input: [h_size_px, v_size_px]
        Output: A, B, C, D, in Autokoordinaten
        """ 
        x_shift = 0
        z_shift = 0
        
        self.drunk = cv.getTrackbarPos("drunk", "control") 
        self.drunk = True #Debug
        if self.drunk == True:
            x_shift = random.randint(-int(Config.street_width_px), int(Config.street_width_px*0.5))
            z_shift = random.randint(-int(Config.street_width_px), int(Config.street_width_px*0.5))
        
        # Berechnung des Punktes A
        a_x = 0 + x_shift
        a_z = -self.h_size_px/2 + z_shift
        A_position = [int(round(a_x, 0)), int(round(a_z, 0))]

        # Berechnung des Punktes B
        b_x = self.v_size_px + x_shift
        b_z = -self.h_size_px/2 + z_shift
        B_position = [int(round(b_x, 0)), int(round(b_z, 0))]

        # Berechnung des Punktes C
        c_x = self.v_size_px + x_shift
        c_z = self.h_size_px/2 + z_shift
        C_position = [int(round(c_x, 0)), int(round(c_z, 0))]

        # Berechnung des Punktes D
        d_x = 0 + x_shift
        d_z = self.h_size_px/2 + z_shift
        D_position = [int(round(d_x, 0)), int(round(d_z, 0))]

        return [A_position, B_position, C_position, D_position] 



    def car_angle(drive_point_coords_list):
        """
        Schritt 2
        Berechne die Drehnung zwischen den Koordinatensystemen K2 und K_hv (Auto und Gesamtbild).
        Input: drive_point_coords_list(Fahrtpunkte des Autos im Bildkoordinatensystem)
        Output: car_angle_list(Winkel phi zwischen dem Autokoordinatensystem und dem Bildkoordinatensystem)
        """

        car_angle_list = []

        # Iteriere über alle außer dem letzten Fahrtpunkt 
        for i in range (0, len(drive_point_coords_list)-1):
           
            # Berechne die x- und z-Differenz zwischen dem aktuellen und dem nächsten Fahrtpunkt
            gegenkathete = drive_point_coords_list[i + 1][1] - drive_point_coords_list[i][1]
            ankathete = drive_point_coords_list[i + 1][0] - drive_point_coords_list[i][0]

            # Berechne den sich daraus ergebeneden Winkel zusammen mit dem Gierwinkel der Kamera gamma
            angle = math.atan(gegenkathete/ankathete) if not ankathete == 0 else math.pi/2
            plus = 90 if ankathete <= 0 else -90
            car_angle_list.append(angle * 180 / math.pi - plus)

        return car_angle_list



    def rotate_img(angle, full_image_nice, full_image_segment, drive_point):
        """
        Schritt 3
        Drehe das Gesamtbild um den Winkel, damit das Auto mit der Schnauze immer direkt nach oben zeigt. 
        Nutze den Fahrtpunkt als Mittelpunkt der Drehung.
        Input: angle(Verdrehwinkel zwischen Koordinatensystem K2 und K_hv), full_image_nice(Gesamtbild realistisch), 
        full_image_segment(Gesamtbild segmentiert), drive_point( Ort wo Auto jetzt steht) 
        Output: full_image_nice(Gesamtbild realistisch und verdreht), full_image_segment(Gesamtbild segmentiert und verdreht)
        """
        [row, cols] = full_image_nice.shape
        drive_point = (drive_point[0], drive_point[1])

        # Erstelle Rotationsmatrix
        rot_mat = cv.getRotationMatrix2D(drive_point, angle, 1.0)

        # Rotiere beide Bilder
        full_image_nice = cv.warpAffine(full_image_nice, rot_mat, (cols, row))
        full_image_segment = cv.warpAffine(full_image_segment, rot_mat, (cols, row))

        return full_image_nice, full_image_segment



    def transform_xz_hv(drive_point, abcd_points_k2):
        """
        Schritt 4
        Ermittle die Punkte (A, B, C, D) im Koordinatensystem K_hv des Gesamtbildes.
        Input: drive_point(Fahrtpunkt des Autos im Bildkoordinatensystem), abcd_points_k2(Punkte im Autokoordinatensystem K2)
        Output: abcd_list(Punkte im Bildkoordinatensystem K_hv)
        """
        abcd_points_hv = []
        o_h = drive_point[0]
        o_v = drive_point[1]

        # Transformation der Koordinatensysteme von K2 in K_hv
        for i in range (0, len(abcd_points_k2)):
           
            k_x = abcd_points_k2[i][0]
            k_z = abcd_points_k2[i][1]

            h_ = k_z + o_h
            v_ = -k_x + o_v

            abcd_points_hv.append([int(h_), int(v_)])

        return abcd_points_hv



    def cut_roi(full_image_nice, full_image_segment, abcd_points_hv):
        """
        Schritt 5
        Schneide das ROI (A, B, C, D) aus dem gedrehten Gesamtbild aus.
        Input: full_image_nice(Gesamtbild realistisch), full_image_segment(Gesamtbild segmentiert), abcd_points_hv (Eckpunkt ABCD in K_hv)
        Output: roi_image_nice(ROI aus realistischen Gesamtbild), roi_image_segment(ROI aus segmentierten Gesamtbild)
        """
        b_h = abcd_points_hv[1][0]
        b_v = abcd_points_hv[1][1]
        a_v = abcd_points_hv[0][1]
        c_h = abcd_points_hv[2][0]

        roi_image_nice = full_image_nice[b_v:a_v, b_h:c_h]
        roi_image_segment = full_image_segment[b_v:a_v, b_h:c_h]

        return roi_image_nice, roi_image_segment



    def transform_perspective(transform_matrix, roi_image_nice, roi_image_segment, camera_resolution):
        """
        Schritt 6
        Transformiere die Perspektive aus dem ROI. Von Vogelperspektive gehe in die Kameraperspektive.
        Input: abcd_list(Punkte im Bildkoordinatensystem),  roi_image_nice(ROI aus realistischen Gesamtbild), 
        roi_image_segment(ROI aus segmentierten Gesamtbild)
        Output: camera_nice(realistische Kameraperspektive), camera_segment(segmentierte Kameraperspektive)
        """

        camera_nice = cv.warpPerspective(roi_image_nice,transform_matrix, camera_resolution, flags = cv.INTER_LINEAR )
        camera_segment = cv.warpPerspective(roi_image_segment,transform_matrix, camera_resolution, flags = cv.INTER_LINEAR )

        return camera_nice, camera_segment




    def add_noise(image, magnitude):
        """
        Diese Funktion überlagert Rauschen auf einem Bild
        Inspiriert von: https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
        Folgendes Rauschen wird überlagert: 
                'gauss'     Gaussian-distributed additive noise.
                'random'    Overlay random noise.
                'sap'       Replaces random pixels with 0 or 1.
        Input: image(Das Originalbild), magnitude(Stärke der Überlagerung, zwischen 0 und 1)
        Output: noisy(verrauschtes Bild)
        """
        image = np.asarray(image)  
        
        row,col = image.shape
        mean = 0
        var = 0.15
        sigma = var**0.5
        s_vs_p = 0.5
        amount = 0.0001 * magnitude        
        
        # Gaußsches Rauschen
        gauss = np.random.normal(mean,sigma,(row, col))
        gauss = gauss.reshape(row,col)      

        # Zufälliges Rauschen
        random = np.random.random((int(col/5), int(row/5))).astype(np.float32)               
        random = cv.resize(random, (col, row))
        
        noisy = image + random * magnitude + gauss * magnitude 
        noisy = np.clip(noisy, 0, 255)
        
        
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy[tuple(coords)] = 255

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy[tuple(coords)] = 0           
         
        
        
        return noisy   



    def add_overlay(image_nice):
        """
        Diese Funktion überlagert das Bild mit Störobjekten.
        Input: 
        image_nice(gesamtes Bild in Vogelperspektive), coords_array(Koordinaten der chunks in Bildkoordinaten)
        Output: 
        image_nice(gesammtes Bild mit Overlay)
        """

        # Wähle ein zufälliges Overlay aus
        number = random.randint(1,85)
        overlay = cv.imread(f"overlays/overlay ({number}).png", cv.IMREAD_UNCHANGED) # Wichtig: Bild muss transparent werden
        
        # Wähle eine zufällige Größe aus
        height = random.randint(10,30)
        width = random.randint(10,25)
        
        overlay = cv.resize(overlay, (width, height), interpolation = cv.INTER_LINEAR)
        overlay_gray = cv.cvtColor(overlay, cv.COLOR_BGR2GRAY)
        
        alpha_channel = overlay[:,:,3]
               
        overlay_mask = alpha_channel * 0
        overlay_mask[alpha_channel > 150] = 1
        
        image_mask = 1 - overlay_mask
        
        # Größe des Kamerabildes
        row,col = image_nice.shape        
        
        # Wähle einen zufälligen Platz für das Overlay
        x_place = random.randint(0, int(row*0.7) - height)
        y_place = random.randint(0, int(col*0.7) - width)
              

        # Addiere die Bilder
        image_nice[x_place : x_place + height, y_place : y_place + width] = overlay_mask * overlay_gray + image_mask * image_nice[x_place : x_place + height, y_place : y_place + width]

        return image_nice
        
        
        
    def add_obstacle(roi_image_nice, roi_image_segment):
        """
        Diese Funktion soll zufällige Hindernisse auf das schöne Kamerabild applizieren und diese entsprechend in der Etikette als Klasse 4 markieren.
        Input:
        image_nice, image_segment
        
        Output:
        image_nice, image_segment               
        """
    
        row,col = roi_image_nice.shape

        # Wähle ein zufälliges Obstacle aus
        number = random.randint(1,34)
        obstacle = cv.imread(f"white_box/box_{number}.jpg", cv.IMREAD_GRAYSCALE) # 
        
        # Wähle eine zufällige Größe aus
        height = random.randint(20,40)
        width = random.randint(30,40)

        obstacle = cv.resize(obstacle, (width, height), interpolation = cv.INTER_LINEAR)

        # Wähle einen zufälligen Platz für das Overlay. Es soll sich nicht zu tief befinden
        x_place = random.randint(0, int(row*0.7) - height)
        y_place = random.randint(0, int(col*0.7) - width)

      
        roi_image_nice[x_place : x_place + height, y_place : y_place + width] = obstacle
        
        obstacle[obstacle > 50] = 250
        roi_image_segment[x_place : x_place + height, y_place : y_place + width, 0] = obstacle      
        
        return roi_image_nice, roi_image_segment
        
        
    def make_metadata(self, camera_nice, camera_segment, path):
        """
        Schritt 7
        Diese Funktion gibt jeder etikettierten Stichprobe einen Index und speichert das schöne und segmentierte Bild ab.
        Außerdem wird eine Datei erstellt, wo später Objektkoordinaten abgelegt werden können.
        Input: 
        self(Unter den Attributen befindet sich der letzte Index), camera_nice(Das verzerrte schöne Bild), 
        camera_segment(Das verzerrte segmentierte Bild), path(Wo sollen die Daten gepeichert werden)
        Output: 
        Die Funktion gibt nichts zurück
        
        """
        self.index += 1

        cv.imwrite(f'data/images/trainset/synthetic_set/image_{self.index}.jpg', camera_nice)
        cv.imwrite(f'data/annotations/trainset/synthetic_set/image_{self.index}.jpg', camera_segment)   
     



    def image_generator(self):
        """
        Diese Funktion generiert eine Stichprobe.
        Input: 
        visualization Objekt
        Output:
        camera_nice, camera_segment (etikettierte Stichprobe)
        """
        
        angle = self.car_angle_list[self.i]
        drive_point = self.drive_point_coords_list[self.i]
        full_image_nice_rot, full_image_segment_rot = Visualization.rotate_img(angle, self.full_image_nice, self.full_image_segment, drive_point)
        abcd_points_hv = Visualization.transform_xz_hv(drive_point, self.abcd_points_k2)
        roi_image_nice, roi_image_segment = Visualization.cut_roi(full_image_nice_rot, full_image_segment_rot, abcd_points_hv)
                
        camera_nice, camera_segment = Visualization.transform_perspective(Config.pre_transform_matrix, roi_image_nice, roi_image_segment, Config.camera_resolution)    
                
        #Config.magnitude = cv.getTrackbarPos("magnitude", "control")/100
        #val_noise = cv.getTrackbarPos("noise", "control")            
        
        # Hier kann man das Sichtfeld der Kamera einschränken       
        half = int(Config.camera_resolution[1]/2)
        full = int(Config.camera_resolution[1])     

        camera_nice = camera_nice[half:full,:]
        camera_segment = camera_segment[half:full,:]
        
        # Reduzierung der Bildgröße
        camera_nice = cv.resize(camera_nice, (Config.input_size_px, int(Config.input_size_px/2)), interpolation = cv.INTER_NEAREST)
        camera_segment = cv.resize(camera_segment, (Config.input_size_px, int(Config.input_size_px/2)), interpolation = cv.INTER_NEAREST)
        
   
        if random.choice([True, False], p=[0.2, 0.8], size=(1))[0]:
            camera_nice, camera_segment = Visualization.add_obstacle(camera_nice, camera_segment)
        
        elif random.choice([True, False], p=[0.4, 0.6], size=(1))[0]:
            camera_nice = Visualization.add_overlay(camera_nice)     
        
        max_val = np.amax(camera_segment, axis = -1)
        camera_segment = np.argmax(camera_segment, axis = -1).astype(np.uint8)*50
        # Was sonst nichts ist, wird Hintergrund
        camera_segment[max_val < 50] = 4*50

        camera_nice = Visualization.add_noise(camera_nice, random.randint(0, 15)) 
        camera_nice = camera_nice * random.randint(80, 100)/100 + random.randint(10, 45)
        camera_nice = np.clip(camera_nice, 0, 255)
        
        
        # Abdeckung des Autos mit einer schwarzen Fläche
        camera_nice = cv.polylines(camera_nice, [Config.points], True, 0, 1)
        camera_nice = cv.fillPoly(camera_nice, [Config.points], 0)
        
        return camera_nice, camera_segment


    def one_datapoint(self, road):        
        """
        Die Funktion führt die obigen Funktionen nacheinander aus.
        Damit kann das Auto entlang der Straße fahren.

        1) Berechne die Punkte ABCD im Koordinatensystem K2 des Autos
        2) Berechne die Drehnung zwischen den Koordinatensystemen K2 und K_hv
        3) Drehe das Gesamtbild um den Winkel. Nutze den Fahrtpunkt als Mittelpunkt
        4) Ermittle die Punkte (A, B, C, D) im Koordinatensystem K_hv
        5) Schneide das ROI (A, B, C, D) aus dem gedrehten Gesamtbild aus
        6) Transformiere die Perspektive aus dem ROI
        7) Speichere jede etikettierte Stichprobe mit Index ab
        """  
        
        if self.i == 0:
            self.full_image_nice, self.full_image_segment, self.drive_point_coords_list, coords_list = Road.build_road(road)  
            
            self.abcd_points_k2 = Visualization.get_ABCD(self)
            self.car_angle_list = Visualization.car_angle(self.drive_point_coords_list)

            camera_nice, camera_segment = Visualization.image_generator(self)
                       
            self.i += 1

        else:
            camera_nice, camera_segment = Visualization.image_generator(self)        
            self.i += 1
            if self.i == len(self.car_angle_list):
                self.i = 0       
                        
    
        return camera_nice, camera_segment
