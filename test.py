import numpy as np
import cv2
import json
import math
import config
import render
import math
from numpy.linalg import inv

with open('config1.json', 'r') as f:
    l=f.read()

config_json = json.loads(l)

if True:
    # Erzeugung der Startlinie
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


c = config.Config("config1.json", debug=False)
r = render.Renderer(c)
r.update_position((1000, 0), math.pi)

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
    bird_view = cv2.warpPerspective(zeros, inv(r.h_segmentation), (3000, 3000), flags=cv2.INTER_NEAREST)
    cv2.imwrite('./bird.png', bird_view)

    
    
