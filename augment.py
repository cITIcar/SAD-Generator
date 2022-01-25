import numpy as np
import cv2
import glob
import os
import random

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
    random = cv2.resize(random, (col, row))

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
    overlay = cv2.imread(f"overlays/overlay ({number}).png", cv2.IMREAD_UNCHANGED) # Wichtig: Bild muss transparent werden

    # Wähle eine zufällige Größe aus
    height = random.randint(10,30)
    width = random.randint(10,25)

    overlay = cv2.resize(overlay, (width, height), interpolation = cv2.INTER_LINEAR)
    overlay_gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)

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
    obstacle = cv2.imread(f"white_box/box_{number}.jpg", cv2.IMREAD_GRAYSCALE) #

    # Wähle eine zufällige Größe aus
    height = random.randint(20,40)
    width = random.randint(30,40)

    obstacle = cv2.resize(obstacle, (width, height), interpolation = cv2.INTER_LINEAR)

    # Wähle einen zufälligen Platz für das Overlay. Es soll sich nicht zu tief befinden
    x_place = random.randint(0, int(row*0.7) - height)
    y_place = random.randint(0, int(col*0.7) - width)


    roi_image_nice[x_place : x_place + height, y_place : y_place + width] = obstacle

    obstacle[obstacle > 50] = 250
    roi_image_segment[x_place : x_place + height, y_place : y_place + width, 0] = obstacle

    return roi_image_nice, roi_image_segment


def augment_dataset(annotations_path, images_path, annotations_output_path, images_output_path, idcs, config):
    
    annotations_list = glob.glob(annotations_path + "/*.jpg")
    images_list = glob.glob(images_path + "/*.jpg")

    if len(annotations_list) == 0:
        print("no annotated png images found under the path", annotations_path)
        return
    if len(images_list) == 0:
        print("no png images found under the path", annotations_path)
        return

    index = 0
    
    #while index < len(idcs):
        #print('run loop')
        
    #TODO Augmentierung kann über index wiederholt werden
    
    
    print("run augment")
    
    for mask_path in annotations_list:
        image_path = mask_path.replace("annotations", "images")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        
        
        #TODO Diese Zeile erzeugt den Fehler

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if mask is None or image is None: # Check if corresponding image exists
            print("skipping invalid file")
            continue

        if random.choice([True, False]):
            row, col = image.shape

            # Wähle ein zufälliges Obstacle aus
            number = random.randint(1, 34)
            obstacle = cv2.imread(f"white_box/box_{number}.jpg", cv2.IMREAD_GRAYSCALE) #

            # Wähle eine zufällige Größe aus
            height = random.randint(20, 40)
            width = random.randint(30, 35)

            obstacle = cv2.resize(obstacle, (width, height), interpolation = cv2.INTER_LINEAR)

            # Wähle einen zufälligen Platz für das Overlay
            x_place = random.randint(0, int(row*0.7) - height)
            y_place = random.randint(0, int(col*0.7) - width)

            overlay_mask = obstacle * 0
            overlay_mask[obstacle > 100] = 1
            image_mask = 1 - overlay_mask

            image[x_place : x_place + height, y_place : y_place + width] = overlay_mask * obstacle + image_mask * image[x_place : x_place + height, y_place : y_place + width]

            space = mask[x_place : x_place + height, y_place : y_place + width]
            space[obstacle > 50] = config["obstacle_class"]
            mask[x_place : x_place + height, y_place : y_place + width] = space

        else:
            image = add_overlay(image)

        image = add_noise(image, random.randint(0, 30))

        cv2.imwrite(f"{annotations_output_path}/image_{index}.png", mask)
        cv2.imwrite(f"{images_output_path}/image_{index}.png", image)

        index = index + 1



