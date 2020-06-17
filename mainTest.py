"""Tests written during developments"""

import random
import time
import traceback
import unittest
from typing import Dict

from main import *


class VisualChecks(unittest.TestCase):

    def test_getNonBackgroundBoxes(self):
        allBoxes = processVideo()

        # print(len(allBoxes))
        try:
            for frameBoxes in allBoxes:
                if not displayBoxes("non background boxes", frameBoxes, reds=[], greens=[]):
                    break
        except:
            traceback.print_exc()
        finally:
            cv2.destroyAllWindows()

    def test_computeAllNeighbors(self):
        allBoxes = processVideo()

        allNeighbors = computeAllNeighbors(allBoxes)

        try:
            for i, frameBoxes in enumerate(allBoxes):
                reds = []
                greens = []

                if i >= 1:

                    pastFrameNeighbors = allNeighbors[i - 1]
                    for pastFrameBoxInd, frameNeighbor in enumerate(pastFrameNeighbors):
                        # green: neighbor in the past
                        greens.append(allBoxes[i - 1][pastFrameBoxInd])


                if not displayBoxes("neighbor computation", frameBoxes, reds=reds, greens=greens):
                    break
        except:
            traceback.print_exc()
        finally:
            cv2.destroyAllWindows()

    def test_loadBackground(self):
        b = loadBackground()
        cv2.imshow("lol", b)
        cv2.waitKey()

    def test_getBlockVels(self):
        allBoxes = processVideo()

        allNeighbors = computeAllNeighbors(allBoxes, bound=30)
        vels = getBlockVels(allBoxes, allNeighbors)
        try:
            displayDirs(vels)
        except:
            traceback.print_exc()
        finally:
            cv2.destroyAllWindows()

    def test_getTrajectories(self):
        allBoxes = processVideo()
        allNeighbors = computeAllNeighbors(allBoxes)

        trajectories = getTrajectories(allBoxes, allNeighbors)
        try:
            if not displayTrajectories(trajectories):
                return
        except:
            traceback.print_exc()
        finally:
            cv2.destroyAllWindows()

    def test_carTrajectories(self):
        allBoxes = processVideo()
        allNeighbors = computeAllNeighbors(allBoxes)

        trajectories = getTrajectories(allBoxes, allNeighbors)

        carTrajectories = list(filter(lambda x: len(x) > 20, trajectories))

        try:
            if not displayTrajectories(carTrajectories):
                return
        except:
            traceback.print_exc()
        finally:
            cv2.destroyAllWindows()

    # pure tampering purpose. don't run it
    def test_drawContour(self):

        with open(os.path.join('cached', 'contours.pickle'), 'rb') as file:
            allContours = pickle.load(file)

        for frameContour in allContours:
            frameContour = list(frameContour.values())

            mask = np.zeros((240, 426), dtype=np.uint8)
            cv2.drawContours(mask, frameContour, -1, 255, -1)

            frame = np.zeros((240, 426, 3), dtype=np.uint8)
            frame[:] = (122, 122, 122)

            final = cv2.bitwise_and(frame, frame, mask=mask)
            cv2.imshow("lol", final)
            k = cv2.waitKey() & 0xff
            if k == 27:
                break

        # print(allContours[1000][0])

    def test_create4PaneDemo(self):
        f1 = np.zeros((240, 426, 3), dtype=np.uint8)
        f2 = np.full((240, 426, 3), 50, dtype=np.uint8)
        f3 = np.full((240, 426, 3), 100, dtype=np.uint8)
        f4 = np.full((240, 426, 3), 200, dtype=np.uint8)

        demoPane = create4PaneDemoFrame([f1, f2, f3, f4], ['1', '2', '3', '4'])
        cv2.imshow("lul", demoPane)
        cv2.waitKey()

    def test_fourPaneDemo(self):
        video = cv2.VideoCapture('trafficVideo.mp4')


        frames = []
        while 1:
            ret, frame = video.read()

            # cv2.imshow("demo", demoFrame)
            if ret:
                frames.append(frame)
            else:
                break
        video.release()

        # shows background subtraction progress and subtracted background
        allBoxes = processVideo()

        allNeighbors = computeAllNeighbors(allBoxes)


        trajectories = getTrajectories(allBoxes, allNeighbors)

        bound = 30
        carTrajectories = list(filter(lambda x: len(x) > bound, trajectories))
        compressed = combineAndWriteAbstraction(carTrajectories, len(allBoxes))


        fourPaneDemo(allBoxes, allNeighbors, trajectories, carTrajectories, frames)
        twoPaneDemo(frames, compressed)


def arrangeintoFrames(trajectories) -> Dict[int, List[List[Tuple]]]:
    frames = {i: [] for i in range(2300)}
    for trajectory in trajectories:
        for timeBox in trajectory:
            i = timeBox[0]
            if i in frames:
                frames[i].append(timeBox[1:])
            else:
                frames[i] = [timeBox[1:]]

    return frames


def displayTrajectories(trajectories):
    frames = arrangeintoFrames(trajectories)

    frameList = list(frames.items())
    frameList.sort(key=lambda x: x[0])
    frameList = list(map(lambda x: x[1], frameList))
    for recs in frameList:
        if not displayBoxes("trajectories", recs, reds=[], greens=[]):
            return False


def displayDirs(dirs):
    velScale = 3

    frame = loadBackground()
    for xInd in range(20):
        for yInd in range(20):
            dir = dirs[xInd][yInd]
            if dir[0] != 0 and dir[1] != 0:
                dir[0] *= velScale
                dir[1] *= velScale
                pt1 = (xInd * (426 / 20) + 426 / 20 / 2, yInd * (240 / 20) + 240 / 20 / 2)
                intPt1 = (int(pt1[0]), int(pt1[1]))
                intPt2 = (int(pt1[0] + dir[0]), int(pt1[1] + dir[1]))
                cv2.arrowedLine(frame, intPt1, intPt2, velToBrightness(dir), thickness=2, tipLength=.5)
    cv2.imwrite("average_velocities.jpg", frame)
    cv2.imshow("dirs", frame)
    cv2.waitKey()


def velToBrightness(vel: Tuple[float, float]):
    # brightnessScale = 3
    sLen = math.sqrt(vel[0] ** 2 + vel[1] ** 2)
    # sLen *= 3
    return [int(255 * math.atan(sLen) / (math.pi / 2))] * 3


def displayBoxes(winname, frameBoxes, delay=0, reds=None, greens=None):
    frame = np.zeros((240, 426, 3), dtype=np.uint8)
    for box in frameBoxes:
        # w h: 426 240
        cv2.rectangle(frame, box[0], box[3], (255, 255, 255))

    for redBox in reds:
        cv2.rectangle(frame, redBox[0], redBox[3], (0, 0, 255), thickness=3)
    for greenBox in greens:
        cv2.rectangle(frame, greenBox[0], greenBox[3], (0, 255, 0), thickness=3)
    cv2.imshow(winname, frame)

    k = cv2.waitKey(delay) & 0xff
    if k == 27:
        return False

    return True


if __name__ == '__main__':
    unittest.main()
