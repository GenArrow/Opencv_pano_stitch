import cv2
import os
import threading
import glob

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def webcam_on():
    global final
    source = cv2.VideoCapture(0)
    win_name = "Preview"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    source.set(12, 30)
    counter = 0
    path = 'd:\\pythonProjectNew\\images\\'


    while True:
        has_frame, frame = source.read()
        if not has_frame:
            break
        cv2.imshow(win_name, frame)
        k = cv2.waitKey(1)
        if k == 27:
            break
        elif k == 32:
            img_name = "frame{}.png".format(counter)
            # frame_copy = frame[int(frame.shape[0]/10):frame.shape[0]-int(frame.shape[0]/10), int(frame.shape[1]/6.9):int(frame.shape[1])-int(frame.shape[1]/6.9)]
            frame_copy = cv2.convertScaleAbs(frame, 0.8, 1)
            cv2.imwrite(os.path.join(path, img_name), frame_copy)
            counter += 1

    try:
        os.remove('d:\\pythonProjectNew\\result.jpg')
    except:
        print("result.jpg not found. Continuing...")
    source.release()
    cv2.destroyWindow(win_name)

    images = []

    for file in glob.glob(path + "**/*.png", recursive=True):
        img = cv2.imread(file)
        images.append(img)
        os.remove(file)

    stitcher = cv2.Stitcher_create()
    status, result = stitcher.stitch(images)
    # result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    # result = result[int(result.shape[0] / 10):result.shape[0] - int(result.shape[0] / 10),
    #         int(result.shape[1] / 10):result.shape[1] - int(result.shape[1] / 10)]
    # cv2.imwrite("result.jpg", result)
    while cv2.waitKey(1) != 27:
        cv2.imshow("Result", result)

        #   shape[1]=width   |    img[a1:a2 , b1:b2], stanga pt height

        resultGray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        ret, threshold = cv2.threshold(resultGray, 0, 255, cv2.THRESH_BINARY)
        # cv2.imshow("Threshold", threshold)

        contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(contours, key=cv2.contourArea)

        mask = np.ones(threshold.shape[:2], dtype="uint8")
        mask = cv2.bitwise_not(mask)

        cv2.rectangle(mask, (0, 0), (mask.shape[1], mask.shape[0]), (0, 0, 0), 2)

        cv2.drawContours(threshold, [cnt], 0, (255, 255, 255), -1)
        # cv2.imshow("ThresholdContour", threshold)

        # cv2.imshow("Mask", mask)

        kernel = np.ones((3, 3), np.uint8)

        while cv2.countNonZero(cv2.subtract(mask, threshold)) > 0:
            mask = cv2.erode(mask, kernel)

        cnts, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntRect = max(cnts, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(cntRect)

        final = result[y:y + h, x:x + w]
        cv2.imshow("Result_Crop", result[y:y + h, x:x + w])

        ## left = tuple(cnt[cnt[:, :, 0].argmin()][0])
        ## right = tuple(cnt[cnt[:, :, 0].argmax()][0])
        ## top = tuple(cnt[cnt[:, :, 1].argmin()][0])
        ## bottom = tuple(cnt[cnt[:, :, 1].argmax()][0])

        ## cv2.imshow("Threshold", threshold)

        ## TC = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)
        ## cv2.circle(TC, left, 8, (0, 50, 255), -1)
        ## cv2.circle(TC, right, 8, (0, 255, 255), -1)
        ## cv2.circle(TC, top, 8, (255, 50, 0), -1)
        ## cv2.circle(TC, bottom, 8, (255, 255, 0), -1)

        ## cv2.imshow("TC", TC)

        # upper_half = threshold[0:int(threshold.shape[0]/2), :]
        # bottom_half = threshold[int(threshold.shape[0]/2):, :]
        # left_half = threshold[:, 0:int(threshold.shape[1]/2)]
        # right_half = threshold[:, int(threshold.shape[1]/2):]
        ## cv2.imshow("U_HALF", upper_half)
        ## cv2.imshow("B_HALF", bottom_half)
        ## cv2.imshow("L_HALF", left_half)
        ## cv2.imshow("R_HALF", right_half)

        # for p_at_height in range(0,upper_half.shape[0]+1):
        #    p_at_width, ok = 0,0
        #    while p_at_width <= upper_half.shape[1]:
        #        if(upper_half[p_at_height, p_at_width]==0):
    cv2.imwrite("result.jpg", final)


t1 = threading.Thread(target=webcam_on())
t1.start()
