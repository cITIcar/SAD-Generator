#!/usr/bin/env python
# coding: utf-8
#from imports import *
import cv2
import numpy as np
from numpy import cos, sin
import math
import time
import road
import sys
import random
import os
import glob
import re

from render_objects import (
    BoxObstacle,
    GaussReflection,
    HiddenGroundRect,
    Dust)

from render import create_h, create_camera_image

import config

RANDOM_CLIP_MAX = True
RANDOM_BRIGHTNESS = True
MAX_DRUNK_X = 200
MAX_DRUNK_Y = 300

random.seed(config.seed)
np.random.seed(config.seed)


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



def create_synthetic(out_path, set_name, camera_points, ground_points, render_objects=[]):
    render_objects.sort(key=lambda obj: obj.ordering)

    ground_points /= config.chunk_size

    # config.render_image_factor increases the resolution of the initial output image and then uses INTER_AREA to reduce
    # losses of lines in the distance
    bird_to_camera_nice = cv2.getPerspectiveTransform(ground_points, camera_points * config.render_image_factor)
    bird_to_camera_segment = cv2.getPerspectiveTransform(ground_points, camera_points)

    r = road.Road()
    idx = 0

    global_angle = 0
    global_position = np.array([0, 0], dtype=np.float64)

    while True:
        t = time.time()
        image, image_segment, drive_points, _ = r.build_road()
        shape = np.array([image.shape[1] * config.chunk_size, image.shape[0] * config.chunk_size])

        for obj in render_objects:
            if obj.pre_transform_step:
                obj.pre_transform_step(
                    image=image,
                    image_segment=image_segment,
                    drive_points=drive_points)

        render_objects.sort(key=lambda obj: obj.ordering)

        drive_points = np.array(drive_points) * config.chunk_size
        angles = car_angle(drive_points)

        drunkness_y = random.random() * MAX_DRUNK_Y
        drunkness_x = random.random() * MAX_DRUNK_X
        drunk_factor_x = np.sin(np.linspace(0, np.pi * random.choice([1, 2]), len(drive_points)))
        drunk_factor_y = np.sin(np.linspace(0, np.pi * random.choice([1, 2]), len(drive_points)))
        brightness_clip = np.sin(np.linspace(0, np.pi * random.choice([1, 2]), len(drive_points)))

        for point, angle, drunk_x, drunk_y, brightness in zip(drive_points, angles, drunk_factor_x, drunk_factor_y, brightness_clip):
            global_position += point - drive_points[0]
            point[0] += drunk_x * drunkness_x
            point[1] += drunk_y * drunkness_y

            perspective_nice = create_camera_image(image, point, angle, bird_to_camera_nice, config.render_image_factor)
            perspective_segment = create_camera_image(image_segment, point, angle, bird_to_camera_segment, 1)

            max_val = np.amax(perspective_segment, axis=-1)
            perspective_segment = np.argmax(perspective_segment, axis=-1).astype(np.uint8) * 50
            perspective_segment[max_val < 50] = 4 * 50

            for obj in render_objects:
                if obj.post_transform_step:
                    obj.post_transform_step(
                        image=perspective_nice,
                        image_segment=perspective_segment,
                        point=point,
                        angle=angle,
                        global_angle=angle + global_angle,
                        bird_to_camera_nice=bird_to_camera_nice,
                        bird_to_camera_segment=bird_to_camera_segment)

            if RANDOM_BRIGHTNESS:
                brighten = np.random.randint(0, 30)
                perspective_nice[perspective_nice < 255 - brighten] += brighten

            if RANDOM_CLIP_MAX: 
                perspective_nice = np.clip(perspective_nice, 0, 155 + brightness * 100)

            perspective_nice = np.clip(perspective_nice, 0, 255).astype(np.uint8)
            perspective_segment = np.clip(perspective_segment, 0, 255).astype(np.uint8)

            cv2.imshow("nice", perspective_nice)
            cv2.imshow("segment", perspective_segment)
            cv2.waitKey(0)

            half_height = int(perspective_nice.shape[0] * 0.45)
            perspective_nice = perspective_nice[half_height: , :]
            perspective_segment = perspective_segment[half_height: , :]

            perspective_nice = cv2.resize(perspective_nice, (config.input_size_px, config.input_size_px // 2), interpolation=cv2.INTER_AREA)
            perspective_segment = cv2.resize(perspective_segment, (config.input_size_px, config.input_size_px // 2), interpolation=cv2.INTER_NEAREST)

            cv2.imwrite(f"{out_path}/images/{set_name}/{idx:03d}.png", perspective_nice)
            cv2.imwrite(f"{out_path}/annotations/{set_name}/{idx:03d}.png", perspective_segment)
            idx += 1

        if angles[-1] > 0:
            global_angle += 90
        elif angles[-1] < 0:
            global_angle -= 90

        total_time = time.time() - t
        total_images = len(drive_points)
        print(f"\033[2A\033[Kfps: {total_images / total_time:.3f} {idx}")
        print(f"position: {global_position}")


def augment_dataset(src_path, out_path):
    index = 0
    for i in range(10):
        for mask_path in glob.glob(src_path):
            image_path = re.sub(r"annotations", "images", mask_path)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if mask is None or image is None:
                continue

            if random.choice([True, False]):
                row,col = image.shape

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
                space[obstacle > 50] = 0
                mask[x_place : x_place + height, y_place : y_place + width] = space
           
            else:
                image = add_overlay(image)
           
            image = add_noise(image, random.randint(0, 30))
            
            cv2.imwrite(f'{out_path}/annotations/image_{index}.png', mask)
            cv2.imwrite(f'{out_path}/images/image_{index}.png', image)
            
            index = index + 1


if __name__ == "__main__":                
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} synthetic [outpath] [setname]")
        print(f"       {sys.argv[0]} augment [srcpath] [outpath] [setname]")
        exit(1)

    if sys.argv[1] == "synthetic":
        out_path = sys.argv[2]
        set_name = sys.argv[3]
        os.makedirs(f"{out_path}/images/{set_name}", exist_ok=True)
        os.makedirs(f"{out_path}/annotations/{set_name}", exist_ok=True)
        render_objects = [
            BoxObstacle(),
            BoxObstacle(),
            HiddenGroundRect(),
            Dust(particles=10000, color=255),
            Dust(particles=50000, color=127),
            Dust(particles=100000, color=63),
            Dust(particles=100000, color=23),
            GaussReflection((200, -400))]

        create_synthetic(out_path, set_name, config.camera_points, config.ground_points, render_objects=render_objects)
    elif sys.argv[1] == "augment":
        if len(sys.argv) < 3:
            print(f"usage: {sys.argv[0]} <synthetic|augment> [srcpath] [outpath]")
            exit(1)
        src_path = sys.argv[2]
        out_path = sys.argv[3]
        os.makedirs(f"{out_path}/images/", exist_ok=True)
        os.makedirs(f"{out_path}/annotations/", exist_ok=True)
        augment_dataset(src_path, out_path)
    else:
        print(f"usage: {sys.argv[0]} <synthetic|augment> [srcpath] [outpath]")

