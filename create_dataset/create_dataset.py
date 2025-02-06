from pyicubsim import iCubGlobalCamera, iCubRightArm, iCubLeftArm, iCubTorso, iCubHead, iCubBall
import numpy as np
import cv2 as cv
from agentspace import space, Agent
import time
import os
import signal
import csv

# exit on ctrl-c
def signal_handler(signal, frame):
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)

#initialize
rightArm = iCubRightArm()
leftArm = iCubLeftArm()
head = iCubHead()
torso = iCubTorso()

# Set a specific position of the arms, torso and head remain unchanged
def set_position(arm_pos):
    arm_pos_tup = tuple(arm_pos)
    rightArm.set(arm_pos_tup)
    leftArm.set(arm_pos_tup)
    time.sleep(1.2)

#Set the iCub robot into a standard position
def set_standard_positions():
    standard_arm_position = (0, 80, 0, 50, 0, 0, 0, 59, 20, 20, 20, 10, 10, 10, 10, 10)
    rightArm.set(standard_arm_position)
    leftArm.set(standard_arm_position)
    head.set((-37.80, 0, -8.800, 0, 0, 0))
    torso.set(joint0=-1, joint1=0.6, joint2=23.20)
    time.sleep(1.5)

set_standard_positions()

# Initialise the CameraAgent class to take pictures, set left eye as the frame
class CameraAgent(Agent):
    def init(self):
        camera = iCubGlobalCamera()
        while True:
            frame = camera.grab()
            space["bgr"] = frame
            cv.imshow("leftEye",frame)
            cv.waitKey(1)

# Initialise the camera agent
cameraAgent = CameraAgent()
time.sleep(1)
dataset = np.loadtxt('dataset.npy')

os.makedirs('images', exist_ok=True)

# Open CSV file in write mode
with open('poses.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['filename', 'label'])

    for i, pose in enumerate(dataset):
        print('pose: ', pose)
        set_position(pose)
        frame = space["bgr"]

        # Create filename
        filename = f'image_{i}.png'
        filepath = os.path.join('images', filename)

        # Save image
        cv.imwrite(filepath, frame)

        # Write entry to CSV
        csv_writer.writerow([filename, pose])

# If the dataset is generated finishes, do the following
print('done')
cameraAgent.stop()
cv.destroyAllWindows()
