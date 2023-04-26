import time
import argparse
from datetime import datetime
from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2
import os
import sys

camera = PiCamera()
camera.resolution = (640, 480)
#camera.rotation = -180
rawCapture = PiRGBArray(camera, size=(640,480))
time.sleep(0.1)

def take_photo(folder_name):
    id = 0
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF
        rawCapture.truncate(0)
        if key == ord("q"):
            sys.exit(0)
        elif key == ord("a"):
            cv2.imwrite(f"{folder_name}/{id}.jpg",image)
            id += 1
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify image IDs")
    parser.add_argument(
        "--folder_name",
        help="Specify the name of the folder to store the images in, to be created if it does not exist",
        default=datetime.now().strftime("%d_%m_%Y__%H_%M_%S"),
        type=str,
        required=False,
    )

    args = parser.parse_args()
    os.makedirs(args.folder_name, exist_ok=True)
    print("Starting camera to take photos")
    take_photo(args.folder_name)
