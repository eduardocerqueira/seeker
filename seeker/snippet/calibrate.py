#date: 2023-04-11T16:52:14Z
#url: https://api.github.com/gists/f02ce6ef839d8cb464bd1a1b482cb258
#owner: https://api.github.com/users/NicerNewerCar

import cv2 as cv
import numpy as np
import glob, os
import click
import yaml

@click.command()
@click.option('--path','-P', help='Path to the calibration images')
@click.option('--checkerboard-size','-CS', default='7x9', help='Size of the pattern')
@click.option('--square-size', '-SS', default=20.0, help='Size of the square in the pattern (mm)')
@click.option('--output', '-O', default='camera_calibration.yaml', help='Output file for the calibration')
def calibrate_camera(path, checkerboard_size,square_size,output):
    print("""
   _____                                  _____      _ _ _               _   _             
  / ____|                                / ____|    | (_) |             | | (_)            
 | |     __ _ _ __ ___   ___ _ __ __ _  | |     __ _| |_| |__  _ __ __ _| |_ _  ___  _ __  
 | |    / _` | '_ ` _ \ / _ \ '__/ _` | | |    / _` | | | '_ \| '__/ _` | __| |/ _ \| '_ \ 
 | |___| (_| | | | | | |  __/ | | (_| | | |___| (_| | | | |_) | | | (_| | |_| | (_) | | | |
  \_____\__,_|_| |_| |_|\___|_|  \__,_|  \_____\__,_|_|_|_.__/|_|  \__,_|\__|_|\___/|_| |_|                    
    """)
    print("Author: Anthony J. Lombardi")
    if path is None:
        raise Exception('Path to the calibration images is required')
    # Validate the path
    if not os.path.exists(path):
        raise Exception('Path does not exist')
    # Validate the checkerboard size
    if len(checkerboard_size.split('x')) != 2:
        raise Exception('Invalid checkerboard size')
    # Validate the square size
    if square_size <= 0:
        raise Exception('Invalid square size')
    # Validate the output file
    if not output.endswith('.yaml'):
        raise Exception('Invalid output file')
    print("Calibrating camera...")
    # Get the checkerboard size
    checkerboard_size = tuple([int(x) for x in checkerboard_size.split('x')])
    # Get the calibration images supported formats: jpg, png, tiff
    images = glob.glob(os.path.join(path, '*.jpg')) + glob.glob(os.path.join(path, '*.png')) + glob.glob(os.path.join(path, '*.tiff'))
    # Create the arrays to store the object points and image points
    objpoints = []
    imgpoints = []
    # Create the object points
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    # Iterate over the images
    for image in images:
        # Read the image
        img = cv.imread(image)
        # Convert to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv.findChessboardCorners(gray, checkerboard_size, None)
        # If corners are found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # Create the dictionary
    data = {'fx': mtx[0, 0], 'fy': mtx[1, 1], 'cx': mtx[0, 2], 'cy': mtx[1, 2], 'k1': dist[0, 0], 'k2': dist[0, 1], 'p1': dist[0, 2], 'p2': dist[0, 3], 'k3': dist[0, 4]}
    # Save the calibration
    with open(output, 'w') as f:
        f.write(yaml.dump(data))
    print("Calibration saved to {}".format(output))
if __name__ == '__main__':
    calibrate_camera()