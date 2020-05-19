import cv2
import numpy as np
import time
import math

def img_capture():
    #webcam = cv2.VideoCapture(1)
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    check, frame = webcam.read()
    '''if count !=0:
        time.sleep(5)
        count+=1'''
    temp = 0
    while temp == 0:
        #cv2.imshow("Capturing", frame)
        cv2.imwrite(filename='saved_img.jpg', img=frame)
        webcam.release()
        img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
        webcam.release()
        temp += 1
        cv2.destroyAllWindows()
def img_distance():
    count2=0
    while(True):
        frame = cv2.imread('saved_img.jpg')
        #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        font = cv2.FONT_HERSHEY_COMPLEX
        Kernal = np.ones((3, 3), np.uint8)
        #lb = np.array([14,79,123])
        #ub = np.array([32,255,255])
        #lb = np.array([21,88,77])
        #ub = np.array([63,255,255])
        #lb = np.array([16,70,107])
        #ub = np.array([60, 255, 255])
        lb = np.array([15, 65, 77])
        ub = np.array([30, 255, 255])

        #mask = cv2.inRange(frame, lb, ub)  ##Create Mask
        mask = cv2.inRange(hsv, lb, ub)
        #cv2.imshow('Masked Image', mask)

        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, Kernal)  ##Morphology
        #cv2.imshow('Opening', opening)

        res = cv2.bitwise_and(frame, frame, mask=opening)  ##Apply mask on original image
        #cv2.imshow('Resuting Image', res)

        contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE,  ##Find contours
                                           cv2.CHAIN_APPROX_NONE)

        if len(contours) != 0:
            cnt = contours[0]
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            area = 3.14 * (radius ** 2)
            #print(area)
            #cv2.circle(frame, center, radius, (0, 255, 0), 2) 220, 71, 36
            cv2.circle(frame, center, radius, (220, 71, 36), 2)
            #cv2.imshow("frame", frame)
            # distance = (16.672 * (area ** -0.356)) # Power Trendline
            #distance = ((-0.546 * (np.log(area))) + 5.3175)  # logarithmic trendline
            #distance = (130.8 * (area ** 2))-(2113* area) +10674
            #distance = ((-14.07 * (np.log(area))) + 147.15)
            distance = ((-18.7 * (np.log(area))) + 211.3)
            if (count2 ==0):
                distance=0
                #print(distance)
                count2+=1
            #dist.append(distance)
            #S = 'Distance Of Object: ' + str(distance)
            S= 'Press Esc Button'
            cv2.putText(frame, S, (5, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.drawContours(frame, cnt, -1, (0, 255, 0), 3)
            #area = cv2.contourArea(cnt)
        else:
            #count2=0
            if (count2 ==0):
                distance=0
                #print(distance)
                count2+=1
        key = cv2.waitKey(1)
        if key == 27:
            break
        cv2.imshow('Original Image', frame)
        # break
    #print(area)
    cv2.destroyAllWindows()
    return distance

# Evecuting the code...

#img_capture()
#d=img_distance()
#print(dd)
