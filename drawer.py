"""helper functions to draw intermediate steps for demo purposes"""

import math
import cv2
from typing import Tuple


def getVelFrame():
    return cv2.imread('average_velocities.jpg')


def getBoxFrames(allBoxes, videoFrames):
    frames = []
    for i, frame in enumerate(videoFrames):
        frame = frame.copy()
        for box in allBoxes[i]:
            # w h: 426 240
            cv2.rectangle(frame, box[0], box[3], (255, 255, 255), thickness=2)

        frames.append(frame)

    return frames


def getNeighborFrames(allBoxes, allNeighbors, videoFrames):
    frames = []
    for i, frame in enumerate(videoFrames):
        frame = frame.copy()

        greens = []

        if i >= 1:

            pastFrameNeighbors = allNeighbors[i - 1]
            for pastFrameBoxInd, frameNeighbor in enumerate(pastFrameNeighbors):
                # green: neighbor in the past
                greens.append(allBoxes[i - 1][pastFrameBoxInd])


        for box in allBoxes[i]:
            # w h: 426 240
            cv2.rectangle(frame, box[0], box[3], (255, 255, 255), thickness=2)

        for green in greens:
            cv2.rectangle(frame, green[0], green[3], (0, 255, 0), thickness=2)

        frames.append(frame)

    return frames



def getTrajectoryFrames(trajectories, videoFrames):

    frames = [frame.copy() for frame in videoFrames]

    counter = 0
    colors = (255,0,0), (0,0,255), (0,255,0)
    for trajectory in trajectories:
        color = colors[counter]

        boxes = []
        frameInds = []
        for timeBox in trajectory:
            frameInds.append(timeBox[0])
            boxes.append(timeBox[1:])

        for i in frameInds:
            for b in boxes:
                cv2.rectangle(frames[i], b[0], b[3], color, thickness=1)

        counter += 1
        counter %= 3

    return frames
