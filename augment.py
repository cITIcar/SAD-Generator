"""Contains functions to augment annotated samples."""

import numpy as np
import cv2
import glob
import random


def add_noise(image, magnitude):
    """
    Add noise to an image.

    Following noise is added to the image:
        'gauss'     Gaussian-distributed additive noise.
        'random'    Overlay random noise.
        'sap'       Replaces random pixels with 0 or 1.

    Parameters
    ----------
    image : Array
        Original image
    magnitude: float
        Magnitude of noise in the range between 0 and 1.

    Returns
    -------
    noisy : Array
        Image with noise
    """
    image = np.asarray(image)

    row, col = image.shape
    mean = 0
    var = 0.15
    sigma = var**0.5
    s_vs_p = 0.5
    amount = 0.0001 * magnitude

    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)

    random = np.random.random((int(col/5), int(row/5))).astype(np.float32)
    random = cv2.resize(random, (col, row))

    noisy = image + random * magnitude + gauss * magnitude
    noisy = np.clip(noisy, 0, 255)

    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[tuple(coords)] = 255

    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1,
                                int(num_pepper)) for i in image.shape]
    noisy[tuple(coords)] = 0

    return noisy


def add_overlay(image):
    """
    Augment image with random overlay.

    Parameters
    ----------
    image : Array
        Image before adding overlay.

    Returns
    -------
    image : Array
        Image with overlay added to it.
    """
    overlay_path = random.choice(glob.glob("overlays/overlay*.png"))
    overlay = cv2.imread(overlay_path,
                         cv2.IMREAD_UNCHANGED)

    height = random.randint(10, 30)
    width = random.randint(10, 25)

    overlay = cv2.resize(overlay, (width, height),
                         interpolation=cv2.INTER_LINEAR)
    overlay_gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)

    alpha_channel = overlay[:, :, 3]

    overlay_label = alpha_channel * 0
    overlay_label[alpha_channel > 150] = 1

    image_label = 1 - overlay_label

    row, col = image.shape

    x_place = random.randint(0, int(row*0.7) - height)
    y_place = random.randint(0, int(col*0.7) - width)

    image[x_place:x_place + height, y_place:y_place + width] = (
        overlay_label * overlay_gray + image_label *
        image[x_place:x_place + height, y_place:y_place + width])

    return image


def add_obstacle(image, label, config):
    """
    Add obstacles to the image and label of a sample.

    The obstacle is randomly choosen from the directory 'white_box'.

    Parameters
    ----------
    image : Array
        Image before adding obstacle.
    label : Array
        label before adding obstacle.
    config : Dict
        Configuration of datagenerator.

    Returns
    -------
    image : Array
        Image with obstacle added to it.
    label: Array
        label with obstacle added to it.
    """
    row, col = image.shape

    obstacle_path = random.choice(glob.glob("obstacles/box_*.jpg"))
    obstacle = cv2.imread(obstacle_path,
                          cv2.IMREAD_GRAYSCALE)

    height = random.randint(20, 40)
    width = random.randint(30, 35)

    obstacle = cv2.resize(obstacle, (width, height),
                          interpolation=cv2.INTER_LINEAR)

    x_place = random.randint(0, int(row*0.7) - height)
    y_place = random.randint(0, int(col*0.7) - width)

    overlay_label = obstacle * 0
    overlay_label[obstacle > 100] = 1
    image_label = 1 - overlay_label

    image[x_place:x_place + height, y_place:y_place + width] = (
        overlay_label * obstacle + image_label *
        image[x_place:x_place + height, y_place:y_place + width])

    space = label[x_place:x_place + height, y_place:y_place + width]
    space[obstacle > 50] = config["obstacle_class"]
    label[x_place:x_place + height, y_place:y_place + width] = space

    return image, label


def augment_dataset(labels_path, images_path, labels_output_path,
                    images_output_path, idcs, config):
    """
    Augment annotated sample.

    This function loads the annotated samples from their source directory,
    augments them by calling augmentation functions and saves them
    into the target directory.

    Parameters
    ----------
    labels_path : String
        Path from which labels of input data will be loaded from.
    images_path : String
        Path from which images of input data will be loaded from.
    labels_output_path : String
        Path where labels of augmented data will be lstored.
    images_output_path : String
        Path where images of augmented data will be lstored.
    idcs : int
        Indices of annotated samples.
    config : Dict
        Configuration of data generator.

    Returns
    -------
    None.
    """
    labels_list = glob.glob(labels_path + "/*.png")
    images_list = glob.glob(images_path + "/*.png")

    if len(labels_list) == 0:
        print("no annotated png images found under the path", labels_path)
        return
    if len(images_list) == 0:
        print("no png images found under the path", labels_path)
        return

    index = 0

    while index < len(idcs):
        for label_path in labels_list:
            image_path = label_path.replace("labels", "images")

            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if label is None or image is None:
                print("skipping invalid file")
                continue

            if random.choice([True, False]):
                image, label = add_obstacle(image, label, config)
            else:
                image = add_overlay(image)

            image = add_noise(image, random.randint(0, 30))

            cv2.imwrite(f"{labels_output_path}/image_{idcs[index]}.png",
                        label)
            cv2.imwrite(f"{images_output_path}/image_{idcs[index]}.png",
                        image)

            index += 1

            if index >= len(idcs):
                break
