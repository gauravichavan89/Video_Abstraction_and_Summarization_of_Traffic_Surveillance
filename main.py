import argparse
import math
import os
import pickle
from sys import stdout

import imageio as imageio
import numpy as np
from typing import List, Tuple, Optional, Dict

import cv2

# def getCarTrajectories(trajectories, vels):
#     carTrajectories = []
#     for trajectory in trajectories:
#         getTrajectoryNormalityScore()
from drawer import getVelFrame, getBoxFrames, getNeighborFrames, getTrajectoryFrames


def generateFreakingGIFs(carTrajectories):
    video = cv2.VideoCapture('trafficVideo.mp4')
    videoFrames = []

    while 1:
        ret, frame = video.read()
        if ret:
            videoFrames.append(frame)
        else:
            break
    video.release()

    allContours: List[Dict]
    with open(os.path.join('cached', 'contours.pickle'), 'rb') as file:
        allContours = pickle.load(file)

    if not os.path.isdir('trajectories'):
        print("trajectories directory not found, generating trajectories")
        os.makedirs('trajectories')

        for i, trajectory in enumerate(carTrajectories):
            trajectoryFrames = []
            print("progress: %d/%d" % (i, len(carTrajectories)))

            for timeBox in trajectory:
                frameID = timeBox[0]
                box = timeBox[1:]

                contour = allContours[frameID][box]

                videoFrame = videoFrames[frameID]

                mask = np.zeros((240, 426), dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)

                final = cv2.bitwise_and(videoFrame, videoFrame, mask=mask)

                trajectoryFrames.append(final[:, :, ::-1])

            imageio.mimsave(os.path.join('trajectories', '%d.gif' % i), trajectoryFrames, duration=1 / 30)
    else:
        print("trajectories directory found, skipping trajectory generation")


def loadContours() -> List[Dict]:
    with open(os.path.join('cached', 'contours.pickle'), 'rb') as file:
        allContours = pickle.load(file)

    return allContours


def loadBackground():
    return cv2.imread(os.path.join('cached', 'background.jpg'))


def loadBoxImages() -> dict:
    with open(os.path.join('cached', 'boxImages.pickle'), 'rb') as file:
        boxImages = pickle.load(file)

    return boxImages


def overlaps(box1, box2) -> bool:
    minx1, maxx1, miny1, maxy1 = box1[0][0], box1[1][0], box1[0][1], box1[2][1]
    minx2, maxx2, miny2, maxy2 = box2[0][0], box2[1][0], box2[0][1], box2[2][1]

    if minx1 > maxx2 or maxx1 < minx2:
        return False
    if miny1 > maxy2 or maxy1 < miny2:
        return False
    return True


def combineAndWriteAbstraction(carTrajectories, length) -> list:
    mergedBoxes = [[] for _ in range(length)]

    background = loadBackground()
    newFrames = [background.copy() for _ in range(length)]

    boxImages = loadBoxImages()

    videoEndInd = 0

    for tInd, t in enumerate(carTrajectories):
        # print(tInd)
        for i in range(len(mergedBoxes)):  # looking for the insertion frame

            offset = t[0][0]

            allFramesCompatible = True
            for timeBox in t:
                tBox = timeBox[1:]
                insertionFrameInd = timeBox[0] - offset + i
                # if insertionFrameInd < 10:
                #     print(insertionFrameInd)
                frameMergedBoxes = mergedBoxes[insertionFrameInd]

                frameIntersects = False

                for mBox in frameMergedBoxes:

                    # print(mBox)
                    # print(tBox)

                    if overlaps(mBox, tBox):
                        # print("y")
                        frameIntersects = True
                        break
                    # print()

                if frameIntersects:
                    allFramesCompatible = False
                    break

            if allFramesCompatible:
                for timeBox in t:
                    box = timeBox[1:]
                    insertionFrameInd = timeBox[0] - offset + i
                    mergedBoxes[insertionFrameInd].append(box)
                    newFrames[insertionFrameInd][box[0][1]:box[2][1], box[0][0]:box[1][0]] = boxImages[timeBox]
                    if insertionFrameInd > videoEndInd:
                        videoEndInd = insertionFrameInd
                break

    height, width, _ = newFrames[0].shape

    video = cv2.VideoWriter('compressed_video.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (width, height))
    for frame in newFrames[:videoEndInd + 1]:
        # cv2.imshow('lol', frame)
        # cv2.waitKey()
        video.write(frame)

    return newFrames[:videoEndInd+1]


def putTextWithCenter(demoFrame, text, center: Tuple[int, int], font_scale: float):
    font = cv2.FONT_HERSHEY_COMPLEX
    thickness = 2
    color = (0, 0, 0)

    size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    cv2.putText(demoFrame, text, (center[0] - size[0] // 2, center[1] - size[1] // 2), font, font_scale, color,
                thickness)


# ALL (ROW, COL)
PADDING = 35
DEMO_SHAPE = (240 * 2 + PADDING * 4, 426 * 2 + PADDING * 4, 3)
CENTER = (240 + 2 * PADDING, 426 + 2 * PADDING)


def createCenteredDemoFrame(frame, title='', footnote=''):
    demoFrame = np.full(DEMO_SHAPE, 255, dtype=np.uint8)
    half_h = frame.shape[0] // 2
    half_w = frame.shape[1] // 2
    demoFrame[CENTER[0] - half_h:CENTER[0] + half_h, CENTER[1] - half_w:CENTER[1] + half_w] = frame
    if title:
        putTextWithCenter(demoFrame, title, (CENTER[1], CENTER[0] - half_h - PADDING // 2), 1.5)
    if footnote:
        putTextWithCenter(demoFrame, footnote, (CENTER[1], CENTER[0] + half_h + 2 * PADDING), 1)
    return demoFrame


def create4PaneDemoFrame(panes, titles=('', '', '', ''), footnote=''):
    demoFrame = np.full(DEMO_SHAPE, 255, dtype=np.uint8)
    h = panes[0].shape[0]
    w = panes[0].shape[1]
    half_h = panes[0].shape[0] // 2
    half_w = panes[0].shape[1] // 2

    demoFrame[PADDING:PADDING + h, PADDING:PADDING + w] = panes[0]
    demoFrame[PADDING:PADDING + h, PADDING * 3 + w:PADDING * 3 + 2 * w] = panes[1]
    demoFrame[PADDING * 3 + h:PADDING * 3 + 2 * h, PADDING:PADDING + w] = panes[2]
    demoFrame[PADDING * 3 + h:PADDING * 3 + 2 * h, PADDING * 3 + w:PADDING * 3 + 2 * w] = panes[3]

    putTextWithCenter(demoFrame, titles[0], (PADDING + half_w, PADDING), 0.8)
    putTextWithCenter(demoFrame, titles[1], (PADDING * 3 + half_w + w, PADDING), 0.8)
    putTextWithCenter(demoFrame, titles[2], (PADDING + half_w, PADDING * 3 + h), 0.8)
    putTextWithCenter(demoFrame, titles[3], (PADDING * 3 + half_w + w, PADDING * 3 + h), 0.8)

    if footnote:
        putTextWithCenter(demoFrame, footnote, (PADDING * 2 + w, PADDING * 4 + 2 * h), 0.8)

    # if title:
    #     putTextWithCenter(demoFrame, title, (CENTER[1], CENTER[0] - half_h - PADDING // 2), 1.5)
    # if footnote:
    #     putTextWithCenter(demoFrame, footnote, (CENTER[1], CENTER[0] + half_h + 2 * PADDING), 1)
    return demoFrame


def demoOriginal() -> list:
    video = cv2.VideoCapture('trafficVideo.mp4')

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    while 1:
        ret, frame = video.read()

        # cv2.imshow("demo", demoFrame)
        if ret:
            frames.append(frame)
        else:
            break
    video.release()

    demoFrame = createCenteredDemoFrame(frames[0], 'Original Video', 'paused, any key to play')
    cv2.imshow("demo", demoFrame)
    cv2.waitKey()

    for frame in frames[1:]:

        demoFrame = createCenteredDemoFrame(frame, 'Original Video', 'ESC to finish video')

        cv2.imshow("demo", demoFrame)
        k = cv2.waitKey(1000 // 30)
        if k & 0xff == 27:
            break

    demoFrame = createCenteredDemoFrame(frames[0], 'Original Video', 'any key to proceed')
    cv2.imshow("demo", demoFrame)
    cv2.waitKey()

    return frames


def fourPaneDemo(allBoxes, allNeighbors, trajectories, carTrajectories, videoFrames):
    boxFrames = getBoxFrames(allBoxes, videoFrames)
    neighborFrames = getNeighborFrames(allBoxes, allNeighbors, videoFrames)
    trajectoryFrames = getTrajectoryFrames(trajectories, videoFrames)
    carTrajectoryFrames = getTrajectoryFrames(carTrajectories, videoFrames)

    # 1228th frame is a good sample

    demoFrame = None
    for i in range(1228, len(videoFrames)):
        demoFrame = create4PaneDemoFrame([boxFrames[i], neighborFrames[i], trajectoryFrames[i], carTrajectoryFrames[i]],
                                         ("1.Non-background boxes",
                                          "2.Neighboring boxes between frames",
                                          "3.Constructed trajectories",
                                          "4.Filter noise by trajectory length"),
                                         "hold any key to play, ESC to proceed")

        cv2.imshow("demo", demoFrame)

        k = cv2.waitKey() & 0xff
        if k == 27:
            break


def twoPaneDemo(videoFrames, compressedFrames):
    for i, frame in enumerate(videoFrames):

        demoFrame = np.full(DEMO_SHAPE, 255, dtype=np.uint8)
        h = frame.shape[0]
        w = frame.shape[1]
        half_h = frame.shape[0] // 2
        half_w = frame.shape[1] // 2

        demoFrame[PADDING * 3 + h:PADDING * 3 + 2 * h, PADDING:PADDING + w] = frame
        if i < len(compressedFrames):
            rightsideFrame = compressedFrames[i]
        else:
            rightsideFrame = np.zeros((240, 426, 3), dtype=np.uint8)

        demoFrame[PADDING * 3 + h:PADDING * 3 + 2 * h, PADDING * 3 + w:PADDING * 3 + 2 * w] = rightsideFrame

        frameDiff = "Frame count: %d -> %d" % (len(videoFrames), len(compressedFrames))
        sizeDiff = "Video size: %.1f MB -> %.1f MB" % (os.path.getsize('trafficVideo.mp4') / 1000000,
                                                 os.path.getsize('compressed_video.mp4') / 1000000)
        putTextWithCenter(demoFrame, 'Result Comparison', (PADDING + w, PADDING), 1)

        putTextWithCenter(demoFrame, frameDiff, (PADDING + w, PADDING+h//3), 0.8)

        putTextWithCenter(demoFrame, sizeDiff, (PADDING + w, PADDING + h*2//3), 0.8)

        putTextWithCenter(demoFrame, 'original', (PADDING + half_w, PADDING * 3 + h), 0.8)
        putTextWithCenter(demoFrame, 'abstracted', (PADDING * 3 + half_w + w, PADDING * 3 + h), 0.8)

        putTextWithCenter(demoFrame, 'ESC to end', (PADDING * 2 + w, PADDING * 4 + 2 * h), 0.8)

        cv2.imshow("demo", demoFrame)
        k = cv2.waitKey(1000//30) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()


def main(args):
    if args.mode == 'abstract':
        assert os.path.isfile('trafficVideo.mp4'), "please put trafficVideo.mp4 under current directory!"

        allBoxes = processVideo()
        # allContours = loadContours()

        allNeighbors = computeAllNeighbors(allBoxes)

        trajectories = getTrajectories(allBoxes, allNeighbors)

        # keep those trajectories that are longer than 1 second (for a 30 fps video)
        bound = 30
        carTrajectories = list(filter(lambda x: len(x) > bound, trajectories))

        print("%d car trajectories found" % len(carTrajectories))

        # generate gifs of individual trajectories
        # generateFreakingGIFs(carTrajectories)

        # get average velocity across the video
        # vels = getBlockVels(allBoxes, allNeighbors)

        combineAndWriteAbstraction(carTrajectories, len(allBoxes))



    else:  # demo
        frames = demoOriginal()

        # shows background subtraction progress and subtracted background
        allBoxes = processVideo(demo=True)

        backgroundFrame = loadBackground()

        allNeighbors = computeAllNeighbors(allBoxes)

        demoFrame = createCenteredDemoFrame(getVelFrame(),
                                            "Average velocity", "press any key to continue")
        cv2.imshow("demo", demoFrame)
        cv2.waitKey()

        trajectories = getTrajectories(allBoxes, allNeighbors)

        bound = 30
        carTrajectories = list(filter(lambda x: len(x) > bound, trajectories))
        compressedFrames = combineAndWriteAbstraction(carTrajectories, len(allBoxes))

        fourPaneDemo(allBoxes, allNeighbors, trajectories, carTrajectories, frames)

        twoPaneDemo(frames, compressedFrames)


def getBlockVels(allBoxes, allNeighbors) -> List[List[List[float]]]:
    """
    divide the frames into 20x20 = 400 square blocks, compute the average velocity vector on each block

    :param allBoxes:
    """

    blockWidth = 426 / 20
    blockHeight = 240 / 20

    movingDir = [[[] for _ in range(20)] for _ in range(20)]
    counters = [[0 for _ in range(20)] for _ in range(20)]

    # block coordinate system, (a, b) are column, row specifically
    for frameInd in range(len(allBoxes) - 1):
        frameBoxes = allBoxes[frameInd]

        for boxInd, box in enumerate(frameBoxes):
            futureNeighborBoxInds = allNeighbors[frameInd][boxInd]

            for futureNeighborBoxInd in futureNeighborBoxInds:
                futureNeighborBox = allBoxes[frameInd + 1][futureNeighborBoxInd]
                for cornerInd, corner in enumerate(futureNeighborBox):
                    blockXInd = min(int(corner[0] // blockWidth), 19)
                    blockYInd = min(int(corner[1] // blockHeight), 19)
                    movingDir[blockXInd][blockYInd].append(
                        np.array([corner[0] - box[cornerInd][0], corner[1] - box[cornerInd][1]]))
                    # movingDir[blockXInd][blockYInd][0] += corner[0] - box[cornerInd][0]
                    # movingDir[blockXInd][blockYInd][1] += corner[1] - box[cornerInd][1]
                    counters[blockXInd][blockYInd] += 1

    for i in range(20):
        for j in range(20):
            # some filtering for this specific case
            if 3 <= len(movingDir[i][j]) and np.std(movingDir[i][j]) < 20:
                movingDir[i][j] = sum(movingDir[i][j]) / counters[i][j]
                # print(movingDir[i][j])
            else:
                movingDir[i][j] = np.array([0, 0])
                # print(movingDir[i][j])
            # if counters[i][j] >= 10:
            #     movingDir[i][j] = sum(movingDir[i][j]) / counters[i][j]
            #     # movingDir[i][j][0] /= counters[i][j]
            #     # movingDir[i][j][1] /= counters[i][j]
            # else:
            #     movingDir[i][j] = np.array([0, 0])
            #     # movingDir[i][j][0] = 0
            #     # movingDir[i][j][1] = 0

    return movingDir


# def pointDis(v1, v2):
#     return math.sqrt((v1[0]-v2[0])**2 + (v1[1] - v2[1])**2)

def computeBoxMSE(box1, box2):
    s = 0
    for i in range(4):
        s += (box1[i][0] - box2[i][0]) ** 2 + (box1[i][1] - box2[i][1]) ** 2

    return math.sqrt(s) / 4


def computeMinNeighborForBox(box, nextFrameBoxes, bound) -> List[int]:
    """
    error := mse(corresponding vertex distance), boxes within an error of bound will be considered neighbors

    :param box:
    :param nextFrameBoxes:
    :param bound:
    """
    neighbors = []
    # counter = 0

    minNeighbor = None
    minMSE = None
    for i, checkingBox in enumerate(nextFrameBoxes):

        mse = computeBoxMSE(box, checkingBox)
        if mse <= bound:

            if minMSE is not None:
                if mse <= minMSE:
                    minMSE = mse
                    minNeighbor = i

            else:
                minMSE = mse
                minNeighbor = i

    if minNeighbor is not None:
        return [minNeighbor]
    else:
        return []


# empirical bound: 30
def computeAllNeighbors(allBoxes, bound=60) -> List[List[List[int]]]:
    allNeighbors = []

    for i in range(len(allBoxes) - 1):
        currentFrameBoxes, nextFrameBoxes = allBoxes[i:i + 2]

        # print(allBoxes[i:i + 2])
        # if nextFrameBoxes:
        #     print(nextFrameBoxes[0])

        frameNeighbors = []
        for box in currentFrameBoxes:
            boxNeighbors = computeMinNeighborForBox(box, nextFrameBoxes, bound)
            frameNeighbors.append(boxNeighbors)
        allNeighbors.append(frameNeighbors)

    return allNeighbors


def getMovementScore(frameInd, boxInd, allBoxes, allNeighbors, vels):
    box = allBoxes[boxInd]
    futureFrameBox = allBoxes[allNeighbors[frameInd][boxInd]]
    total = 0
    while futureFrameBox:
        pass


def getTrajectories(allBoxes, allNeighbors):
    trajectories = []
    trajectoryHeads = dict()

    # includedBoxeInds = set()

    for frameInd in range(len(allBoxes) - 1):
        frameBoxes = allBoxes[frameInd]
        for boxInd, box in enumerate(frameBoxes):

            timeBox = (frameInd,) + box
            if allNeighbors[frameInd][boxInd]:

                futureTimeBox = (frameInd + 1,) + allBoxes[frameInd + 1][allNeighbors[frameInd][boxInd][0]]
                if timeBox in trajectoryHeads:
                    trajectories[trajectoryHeads[timeBox]].append(futureTimeBox)
                    trajectoryHeads[futureTimeBox] = trajectoryHeads[timeBox]
                    trajectoryHeads.pop(timeBox)
                else:
                    trajectories.append([timeBox, futureTimeBox])
                    trajectoryHeads[futureTimeBox] = len(trajectories) - 1

    return trajectories


def roadAndSizeDetection(boxes):
    return None, None


def retrieveComputed() -> Tuple[Optional[List[List[Tuple]]], bool]:
    file = os.path.join('cached', 'trafficVideoBoxes.pickle')
    contourFile = os.path.join('cached', 'contours.pickle')
    backgroundImg = os.path.join('cached', 'background.jpg')
    boxImages = os.path.join('cached', 'boxImages.pickle')

    if os.path.isfile(file) and os.path.isfile(contourFile) and os.path.isfile(backgroundImg) and \
            os.path.isfile(boxImages):
        print("cached result found, skipping GMG background detection")
        with open(file, 'rb') as file:
            allBoxes = pickle.load(file)
        return allBoxes, False
    else:
        print("cached result not found, proceeding to GMG background detection")
        return None, True


def processVideo(demo=False) -> List[List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]]]:
    """
    Get non-background bounding boxes every frame by GMG algorithm.
    Some initial filtering is used to filter boxes too small, (smaller than 100 square pixels)

    in the meantime, both background and contours are extracted and cached for later use.

    :return: boxes
    """
    demoFrame = np.full((240 * 2, 426 * 2, 3), 255, dtype=np.uint8)

    allContrours = []

    allBoxes, caching = retrieveComputed()
    if allBoxes is None or demo:
        allBoxes = []
        caching = True
    else:
        return allBoxes

    assert os.path.isfile('trafficVideo.mp4'), "trafficVideo.mp4 not found"

    video = cv2.VideoCapture('trafficVideo.mp4')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    boxImages = dict()

    backgroundFrequencies = [[dict() for _ in range(width)] for _ in range(height)]
    background = np.full((240, 426, 3), 255, dtype=np.uint8)
    counter = 0

    while 1:
        ret, frame = video.read()

        print("Video processing progress: %d\r" % ((counter + 1) * 100 / length), end="")
        if demo:
            footnote = ''
            if 300 <= counter <= 320:
                footnote = 'Sampling background...'

                for r in range(height):
                    for c in range(width):
                        colorFreqs = tuple(backgroundFrequencies[r][c].items())
                        if colorFreqs:
                            maxP = max(colorFreqs)[0]
                            background[r][c] = maxP

                showingFrame = background
            else:
                showingFrame = np.full((240, 426, 3), 255, dtype=np.uint8)

            demoFrame = createCenteredDemoFrame(showingFrame,
                                                "Processing Video: %d%%" % (
                                                        (counter + 1) * 100 / length), footnote)
            cv2.imshow("demo", demoFrame)
            cv2.waitKey(1)

        if ret:
            frameBoxes = []
            frameContours = dict()

            fgmask = fgbg.apply(frame)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            _, th1 = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
            _, contours, _ = cv2.findContours(th1, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)  # showing the masked result

            # if counter // 50 == 0:
            if 300 <= counter <= 320:
                # stdout.flush()
                bgmask = np.logical_not(fgmask)

                # cv2.waitKey()
                bg = cv2.bitwise_and(frame, frame, mask=bgmask.astype(np.uint8))
                # cv2.imshow("lul", bg)
                for r in range(height):
                    for c in range(width):
                        if bgmask[r][c]:
                            p = tuple(bg[r][c])
                            k = backgroundFrequencies[r][c]
                            if p in k:
                                k[p] += 1
                            else:
                                k[p] = 1

            for i in range(len(contours)):
                if len(contours[i]) >= 5:

                    # geting the 4 points of rectangle

                    x, y, w, h = cv2.boundingRect(contours[i])
                    if w * h >= 100:
                        # upper-left upper-right lower-left lower-right
                        box = ((x, y), (x + w, y), (x, y + h), (x + w, y + h))
                        frameBoxes.append(box)
                        boxImages[(counter,) + box] = frame[y:y + h, x:x + w]
                        frameContours[box] = contours[i]
            allContrours.append(frameContours)
            allBoxes.append(frameBoxes)

        else:
            break
        counter += 1
    print("Video processing progress: 100")

    video.release()

    if not demo:
        for r in range(height):
            for c in range(width):
                maxP = max(tuple(backgroundFrequencies[r][c].items()), key=lambda x: x[1])[0]
                background[r][c] = maxP

    if caching:
        if not os.path.exists('cached'):
            os.makedirs('cached')
        filename = os.path.join('cached', 'trafficVideoBoxes.pickle')

        contoursFilename = os.path.join('cached', 'contours.pickle')
        backgroundFilename = os.path.join('cached', 'background.jpg')
        boxImagesFilename = os.path.join('cached', 'boxImages.pickle')

        with open(filename, 'wb') as file:
            pickle.dump(allBoxes, file)

        with open(contoursFilename, 'wb') as file:
            pickle.dump(allContrours, file)

        with open(boxImagesFilename, 'wb') as file:
            pickle.dump(boxImages, file)

        # print(background[0:10, 0:10])
        cv2.imwrite(backgroundFilename, background)

        print("bounding boxes, contours, extracted background cached for later use")

        print("video frame count: %d" % length)

    return allBoxes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simplify traffic monitoring videos')

    parser.add_argument('-m', '--mode', default='abstract', type=str, required=True, choices=('abstract', 'demo'),
                        help='Enter abstract to do video abstraction. Enter demo to show intermediate steps also.')

    main(parser.parse_args())
