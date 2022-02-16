"""
Augmentation of real world images with start lines.
This GUI allows to transform real world camera images in bird's-eye-view and add
a start line inside it. After the adding of the startline, the image will be transformed back into 
camera perspective.
"""

import numpy as np
import cv2
import json
import math
import config
import render
import math
import glob
import imutils
from numpy.linalg import inv


class Startline:

with open('config1.json', 'r') as f:
    l=f.read()

config_json = json.loads(l)

translation_step = 25
scale_step = 25
rotation_step = 30

# Define render objects
c = config.Config("config1.json", debug=False)
r = render.Renderer(c)
r.update_position((1400, 0), math.pi)

# Create Startline
start_line_rows = config_json["start_line_rows"]
start_line_colums = config_json["start_line_colums"]
patch_size = config_json["patch_size"]

white_patch = np.random.randint(150, 255, (patch_size, patch_size))


start_line_image = np.random.randint(0, 50, (patch_size*start_line_rows, patch_size*start_line_colums))
start_line_mask = np.ones((patch_size*start_line_rows, patch_size*start_line_colums))*255

for i in range(start_line_rows):
    y_start = i * patch_size
    y_end = (i + 1)*patch_size
    
    for j in range(start_line_colums):
        if 2*j+1+i%2 > start_line_colums:
            break
        x_start = (2*j+i%2)*patch_size
        x_end = (2*j+1+i%2)*patch_size
        start_line_image[y_start:y_end, x_start:x_end] = white_patch

"""
cv2.imshow("startline", start_line_image.astype(np.uint8))
cv2.waitKey(0) 
cv2.destroyAllWindows() 
"""


angle = 0
offset_x = 1000
offset_y = 1000
scale = 1
k = 0
forward = False


# Read all real images
annotations_list = glob.glob("./real_dataset/annotations/set_*/*.jpg")

if len(annotations_list) == 0:
    print("no annotated images found under the path", annotations_path)

for mask_path in annotations_list:
    image_path = mask_path.replace("annotations", "images")

    camera_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    camera_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


    zeros_image = np.zeros((3000, 3000))
    zeros_mask = np.zeros((3000, 3000))

    zeros_image[offset_x:offset_x+patch_size*start_line_rows, offset_y:offset_y+patch_size*start_line_colums] = start_line_image
    zeros_mask[offset_x:offset_x+patch_size*start_line_rows, offset_y:offset_y+patch_size*start_line_colums] = 250
    
    interpolation = cv2.INTER_LINEAR  # cv2.INTER_NEAREST

    # Transform to bird's eye view
    zeros_1 = np.zeros((128, 128))
    zeros_2 = np.zeros((128, 128))
    zeros_1[64:128, :] = camera_mask
    zeros_2[64:128, :] = camera_image
    zeros_1_large = cv2.resize(zeros_1, (640, 480), interpolation = interpolation)
    zeros_2_large = cv2.resize(zeros_2, (640, 480), interpolation = interpolation)
    bird_mask = cv2.warpPerspective(zeros_1_large, inv(r.h_segmentation), (3000, 3000), flags=interpolation)
    bird_image = cv2.warpPerspective(zeros_2_large, inv(r.h_segmentation), (3000, 3000), flags=interpolation)


    while(True):

        if(k == ord("x")):
            zeros_image = imutils.translate(zeros_image, (translation_step, 0))
            zeros_mask = imutils.translate(zeros_mask, (translation_step, 0))

        elif(k == ord("y")):
            zeros_image = imutils.translate(zeros_image, (-translation_step, 0))
            zeros_mask = imutils.translate(zeros_mask, (-translation_step, 0))

            
        if(k == ord("d")):
            zeros_image = imutils.translate(zeros_image, (0, translation_step))
            zeros_mask = imutils.translate(zeros_mask, (0, translation_step))
            
        elif(k == ord("a")):
            zeros_image = imutils.translate(zeros_image, (0, -translation_step))
            zeros_mask = imutils.translate(zeros_mask, (0, -translation_step))
            
        if(k == ord("w")):
            
            pass
            
        elif(k == ord("s")):
            
            pass

        if(k == ord("r")):
            zeros_image = imutils.rotate(zeros_image, rotation_step)
            zeros_mask = imutils.rotate(zeros_mask, rotation_step)

            
        elif(k == ord("e")):
            zeros_image = imutils.rotate(zeros_image, -rotation_step)
            zeros_mask = imutils.rotate(zeros_mask, -rotation_step)


        

        # Transform to camera view
        result_camera_mask_large = cv2.warpPerspective(zeros_mask, r.h_segmentation, (640, 480), flags=interpolation)
        result_camera_image_large = cv2.warpPerspective(zeros_image, r.h_segmentation, (640, 480), flags=interpolation)
        result_camera_mask = cv2.resize(result_camera_mask_large, (128, 128), interpolation = interpolation)
        result_camera_image = cv2.resize(result_camera_image_large, (128, 128), interpolation = interpolation)
        result_camera_mask = result_camera_mask[64:128, :]
        result_camera_image = result_camera_image[64:128, :]

        
        bird_mask_clipped = np.clip(bird_mask + zeros_mask, 0, 255)
        bird_image_clipped = np.clip(bird_image + zeros_image, 0, 255)

        bird_mask_resized = cv2.resize(bird_mask_clipped, (1000, 1000), interpolation = interpolation)
        bird_image_resized = cv2.resize(bird_image_clipped, (1000, 1000), interpolation = interpolation)
        bird_mask_half = bird_mask_resized[0:500, :]
        bird_image_half = bird_image_resized[0:500, :]

        result_camera_mask = cv2.resize(result_camera_mask, (1000, 500), interpolation = interpolation)
        result_camera_image = cv2.resize(result_camera_image, (1000, 500), interpolation = interpolation)

        result_mask = np.concatenate([bird_mask_half, result_camera_mask], axis=0)
        result_image = np.concatenate([bird_image_half, result_camera_image], axis=0)
        result = np.concatenate([result_mask, result_image], axis=1)

        cv2.imshow("result", result.astype(np.uint8))

        while True:
            k = cv2.waitKey(0)
            if k == 27:
                break    
            if k == ord(" "):
                exit()
            

cv2.destroyAllWindows()

    

if False:
    # Erzeugung der Startlinie
    # Die Straße ist in Vogelperspektive 40cm breit
    start_line_rows = config_json["start_line_rows"]
    start_line_colums = config_json["start_line_colums"]
    patch_size = config_json["patch_size"]
    white = np.random.randint(150, 255, (patch_size, patch_size))
    white_patch = np.zeros((patch_size, patch_size, 4))
    white_patch[:,:,0] = white
    white_patch[:,:,1] = white
    white_patch[:,:,2] = white
    white_patch[:,:,3] = np.random.randint(230, 255, (patch_size, patch_size))
    img = np.random.randint(0, 50, (patch_size*start_line_rows, patch_size*start_line_colums, 4))

    for i in range(start_line_rows):
        y_start = i * patch_size
        y_end = (i + 1)*patch_size
        
        for j in range(start_line_colums):
            if 2*j+1+i%2 > start_line_colums:
                break
            x_start = (2*j+i%2)*patch_size
            x_end = (2*j+1+i%2)*patch_size
            img[y_start:y_end, x_start:x_end, :] = white_patch

    cv2.imwrite("./startline.png", img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 



"""
TODO Schwarze Fläche der Startlinie transparent machen

"""

# Erstellung der Kameraperspektive
if False:
    img = cv2.imread('./test.png', 0)   
    zeros = np.zeros((128, 128))
    zeros[64:128, :] = cv2.imread('./real_dataset/images/set_1/image_0.jpg', 0) 
    zeros = cv2.resize(zeros, (640, 480), interpolation = cv2.INTER_NEAREST)
    cv2.imwrite('./original.png', zeros) 
    
    bird_view = cv2.warpPerspective(img, r.h_segmentation, (640, 480), flags=cv2.INTER_NEAREST)
    cv2.imwrite('./camera_startline.png', bird_view)    
    
    
# Erstellung der Vogelperspektive    
if False: 

    img = cv2.imread('./real_dataset/images/set_1/image_0.jpg', 0)   
    zeros = np.zeros((128, 128))
    zeros[64:128, :] = img
    zeros = cv2.resize(zeros, (640, 480), interpolation = cv2.INTER_NEAREST)
    bird_view = cv2.warpPerspective(zeros, inv(r.h_segmentation), (2000, 2000), flags=cv2.INTER_NEAREST)
    cv2.imwrite('./bird.png', bird_view)

if __name__ == "__main__":
   
    
