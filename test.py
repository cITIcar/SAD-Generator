import numpy as np
import cv2
import json
import math
import config
import render
import math
import glob
from numpy.linalg import inv

with open('config1.json', 'r') as f:
    l=f.read()

config_json = json.loads(l)




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
offset_x = 0
offset_y = 0
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

    # Define render objects
    c = config.Config("config1.json", debug=False)
    r = render.Renderer(c)
    r.update_position((1400, 0), math.pi)

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

    # Insert Startline
    # Arrowkey translate startline
    # a/d rotate startline
    # w/s scale startline
    
    print(k)
    if():
        scale += 0.1
        start_line_image = cv2.resize(start_line_image, (patch_size*start_line_rows*scale, patch_size*start_line_colums*scale), interpolation = interpolation)
        start_line_mask = cv2.resize(start_line_mask, (patch_size*start_line_rows*scale, patch_size*start_line_colums*scale), interpolation = interpolation)
    elif():
        scale -= 0.1
        start_line_image = cv2.resize(start_line_image, (patch_size*start_line_rows*scale, patch_size*start_line_colums*scale), interpolation = interpolation)
        start_line_mask = cv2.resize(start_line_mask, (patch_size*start_line_rows*scale, patch_size*start_line_colums*scale), interpolation = interpolation)
        
    if():
        offset_x += 5
        
    elif():
        offset_x -= 5
        
    if():
        offset_y += 5
        
    elif():
        offset_y -= 5

    if():
        angle += 5
        
    elif():
        angle -= 5


    # Transform to camera view
    result_camera_mask_large = cv2.warpPerspective(bird_mask, r.h_segmentation, (640, 480), flags=interpolation)
    result_camera_image_large = cv2.warpPerspective(bird_image, r.h_segmentation, (640, 480), flags=interpolation)
    result_camera_mask = cv2.resize(result_camera_mask_large, (128, 128), interpolation = interpolation)
    result_camera_image = cv2.resize(result_camera_image_large, (128, 128), interpolation = interpolation)
    result_camera_mask = result_camera_mask[64:128, :]
    result_camera_image = result_camera_image[64:128, :]

    bird_mask = cv2.resize(bird_mask, (1000, 1000), interpolation = interpolation)
    bird_image = cv2.resize(bird_image, (1000, 1000), interpolation = interpolation)
    bird_mask = bird_mask[0:500, :]
    bird_image = bird_image[0:500, :]

    result_camera_mask = cv2.resize(result_camera_mask, (1000, 500), interpolation = interpolation)
    result_camera_image = cv2.resize(result_camera_image, (1000, 500), interpolation = interpolation)

    result_mask = np.concatenate([bird_mask, result_camera_mask], axis=0)
    result_image = np.concatenate([bird_image, result_camera_image], axis=0)
    result = np.concatenate([result_mask, result_image], axis=1)

    cv2.imshow("result", result.astype(np.uint8))

    k = cv2.waitKey(0)
    if k == 27:
        break

    while forward == False:
        k = cv2.waitKey(0)
        if k == 27:
            forward = True
    
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

    
    
