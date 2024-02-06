#!/usr/bin/env python3
import cv2
import json
from functools import partial
from colorama import Fore

# ---------------------------------
# functions called in trackbar event
# ---------------------------------

# Trackbar min Blue

def onTrackbarBmin(threshold, image, window_name, Limites, trackbar_name_Bmax):
    # update dict Limites with the new value of minimum Blue
    Limites['B']['min'] = threshold

    # code to prevent that minimum value reaches maximum value
    if threshold > Limites['B']['max']:
        cv2.setTrackbarPos(trackbar_name_Bmax, window_name, threshold + 1)

    # calculation of mask image using the new limits
    mask = cv2.inRange(image, (Limites['B']['min'], Limites['G']['min'], Limites['R']['min']),
                       (Limites['B']['max'], Limites['G']['max'], Limites['R']['max']))


# Trackbar max Blue
def onTrackbarBmax(threshold, image, window_name, Limites, trackbar_name_Bmin):
    # update dict Limites with the new value of maximum Blue
    Limites['B']['max'] = threshold

    # code to prevent that minimum value reaches maximum value
    if threshold < Limites['B']['min']:
        cv2.setTrackbarPos(trackbar_name_Bmin, window_name, threshold - 1)

    # calculation of mask image using the new limits
    mask = cv2.inRange(image, (Limites['B']['min'], Limites['G']['min'], Limites['R']['min']),
                       (Limites['B']['max'], Limites['G']['max'], Limites['R']['max']))


# Trackbar min Blue
def onTrackbarGmin(threshold, image, window_name, Limites, trackbar_name_Gmax):
    # update dict Limites with the new value of minimum Green
    Limites['G']['min'] = threshold

    # code to prevent that minimum value reaches maximum value
    if threshold > Limites['G']['max']:
        cv2.setTrackbarPos(trackbar_name_Gmax, window_name, threshold + 1)

    # calculation of mask image using the new limits
    mask = cv2.inRange(image, (Limites['B']['min'], Limites['G']['min'], Limites['R']['min']),
                       (Limites['B']['max'], Limites['G']['max'], Limites['R']['max']))


# Trackbar min Blue
def onTrackbarGmax(threshold, image, window_name, Limites, trackbar_name_Gmin):
    # update dict Limites with the new value of maximum Green
    Limites['G']['max'] = threshold

    # code to prevent that minimum value reaches maximum value
    if threshold < Limites['G']['min']:
        cv2.setTrackbarPos(trackbar_name_Gmin, window_name, threshold - 1)

    # calculation of mask image using the new limits
    mask = cv2.inRange(image, (Limites['B']['min'], Limites['G']['min'], Limites['R']['min']),
                       (Limites['B']['max'], Limites['G']['max'], Limites['R']['max']))


# Trackbar min Blue
def onTrackbarRmin(threshold, image, window_name, Limites, trackbar_name_Rmax):
    # update dict Limites with the new value of minimum Red
    Limites['R']['min'] = threshold

    # code to prevent that minimum value reaches maximum value
    if threshold > Limites['R']['max']:
        cv2.setTrackbarPos(trackbar_name_Rmax, window_name, threshold + 1)

    # calculation of mask image using the new limits
    mask = cv2.inRange(image, (Limites['B']['min'], Limites['G']['min'], Limites['R']['min']),
                       (Limites['B']['max'], Limites['G']['max'], Limites['R']['max']))


# Trackbar min Blue
def onTrackbarRmax(threshold, image, window_name, Limites, trackbar_name_Rmin):
    # update dict Limites with the new value of maximum Red
    Limites['R']['max'] = threshold

    # code to prevent that minimum value reaches maximum value
    if threshold < Limites['R']['min']:
        cv2.setTrackbarPos(trackbar_name_Rmin, window_name, threshold - 1)

    # calculation of mask image using the new limits
    mask = cv2.inRange(image, (Limites['B']['min'], Limites['G']['min'], Limites['R']['min']),
                       (Limites['B']['max'], Limites['G']['max'], Limites['R']['max']))


# -----------------------------------------
# End of functions called in trackbar event
# -----------------------------------------

def main():
    # -------------
    # Initial Setup
    # -------------

    window_name = "Color Segmenter"
    cv2.namedWindow(window_name)

    # # choosing the camera
    # capture = cv2.VideoCapture(0)

    # # first capture of camera, needed for the creation of trackbars
    # _, image = capture.read()
    file_index = 9
    file_prefix = f"{file_index:02}"
    filename_rgb_image = f'rgbd_scenes_dataset/{file_prefix}-color.png'
    image = cv2.imread(filename_rgb_image)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    

    # Creating the dict that contain all information regarding the segmentation limits
    Limites = {'B': {'max': 255, 'min': 0}, 'G': {'max': 255, 'min': 0}, 'R': {'max': 255, 'min': 0}}

    # trackbar's names
    trackbar_name_Bmin = 'Min B'
    trackbar_name_Bmax = 'Max B'
    trackbar_name_Gmin = 'Min G'
    trackbar_name_Gmax = 'Max G'
    trackbar_name_Rmin = 'Min R'
    trackbar_name_Rmax = 'Max R'

    # creation of partial functions that enabled us not using global variables
    myonTrackbarBmin = partial(onTrackbarBmin, image=hsv_image, window_name=window_name, Limites=Limites,
                               trackbar_name_Bmax=trackbar_name_Bmax)
    myonTrackbarBmax = partial(onTrackbarBmax, image=hsv_image, window_name=window_name, Limites=Limites,
                               trackbar_name_Bmin=trackbar_name_Bmin)
    myonTrackbarGmin = partial(onTrackbarGmin, image=hsv_image, window_name=window_name, Limites=Limites,
                               trackbar_name_Gmax=trackbar_name_Gmax)
    myonTrackbarGmax = partial(onTrackbarGmax, image=hsv_image, window_name=window_name, Limites=Limites,
                               trackbar_name_Gmin=trackbar_name_Gmin)
    myonTrackbarRmin = partial(onTrackbarRmin, image=hsv_image, window_name=window_name, Limites=Limites,
                               trackbar_name_Rmax=trackbar_name_Rmax)
    myonTrackbarRmax = partial(onTrackbarRmax, image=hsv_image, window_name=window_name, Limites=Limites,
                               trackbar_name_Rmin=trackbar_name_Rmin)

    # creation of 6 trackbar that will enabled the user change the segmentation limits
    cv2.createTrackbar(trackbar_name_Bmin, window_name, 0, 255, myonTrackbarBmin)
    cv2.createTrackbar(trackbar_name_Bmax, window_name, 0, 255, myonTrackbarBmax)
    cv2.createTrackbar(trackbar_name_Gmin, window_name, 0, 255, myonTrackbarGmin)
    cv2.createTrackbar(trackbar_name_Gmax, window_name, 0, 255, myonTrackbarGmax)
    cv2.createTrackbar(trackbar_name_Rmin, window_name, 0, 255, myonTrackbarRmin)
    cv2.createTrackbar(trackbar_name_Rmax, window_name, 0, 255, myonTrackbarRmax)

    # Set initial segmentation limits
    cv2.setTrackbarPos(trackbar_name_Bmin, window_name, 0)
    cv2.setTrackbarPos(trackbar_name_Bmax, window_name, 255)
    cv2.setTrackbarPos(trackbar_name_Gmin, window_name, 0)
    cv2.setTrackbarPos(trackbar_name_Gmax, window_name, 255)
    cv2.setTrackbarPos(trackbar_name_Rmin, window_name, 0)
    cv2.setTrackbarPos(trackbar_name_Rmax, window_name, 255)

    # program's cycle that will capture the camera's image, calculate the mask using updated segmentation limits and
    # showing the mask image to the user
    while True:
        # _, image = capture.read()
        image = cv2.imread(filename_rgb_image)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, (Limites['B']['min'], Limites['G']['min'], Limites['R']['min']),
                           (Limites['B']['max'], Limites['G']['max'], Limites['R']['max']))
        cv2.imshow(window_name, mask)
        cv2.imshow('Real Image', image)

        key = cv2.waitKey(1)
        if key == 'w' or key == 'q':
            break

    if key == 'w':
        file_name = 'limits.json'
        with open(file_name, 'w') as file_handle:
            print(Fore.GREEN + 'writing dictionary limits to file ' + file_name + Fore.RESET)
            json.dump(Limites, file_handle)

    # ---------------------
    # finishing the program
    # ---------------------
    cv2.destroyAllWindows()
    # capture.release()


if __name__ == '__main__':
    main()
