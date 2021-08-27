import numpy as np
import cv2
import pyautogui
from cv2 import FONT_HERSHEY_SIMPLEX

####################################################### DEFINING COLOR RANGE ###################################################
green_lower = np.array([36, 25, 25])
green_upper = np.array([70, 255,255])

blue_lower = np.array([78,158,124])
blue_upper = np.array([138,255,255])

skin_lower = np.array([0,133,77])
skin_upper = np.array([235,173,127])


cap = cv2.VideoCapture(0)
prev_y = 0
closed_fist = cv2.CascadeClassifier('closed_palm.xml')          #used for recognition of closed palm
finger_tip = cv2.CascadeClassifier('open_palm.xml')             #used for recognition of open palm


while True:
    ret, frame = cap.read()

    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    hand_close = closed_fist.detectMultiScale(gray,1.3,5)
    finger = finger_tip.detectMultiScale(gray, 1.3, 5)


####################################################### MASKING ###################################################
    mask_for_down = cv2.inRange(hsv, green_lower, green_upper)
    mask_for_up = cv2.inRange(hsv, blue_lower, blue_upper)

    contours_for_down, hierarchy_for_down = cv2.findContours(mask_for_down, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_for_up, hierarchy_for_up = cv2.findContours(mask_for_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


####################################################### FOR OPENING OF FOLDER ###################################################
    for x,y,w,h in finger:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, 'Opening', (50, 50), FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
        pyautogui.press('enter',presses =1)
        break


####################################################### FOR CLOSING OF FOLDER ###################################################
    for x,y,w,h in hand_close:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255, 0, 0),2)
        cv2.putText(frame, 'Closing', (50, 50), FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
        pyautogui.press('backspace',presses =1)
        break


####################################################### FOR SCROLLING DOWN ###################################################
    for c_down in contours_for_down:
        area = cv2.contourArea(c_down)
        if area>300:
            x,y,w,h = cv2.boundingRect(c_down)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            if y < prev_y-40:
                cv2.putText(frame, 'Scrolling down', (50, 50), FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
                pyautogui.press('down',presses=3)
            prev_y = y


####################################################### FOR SCROLLING UP ###################################################
    for c_up in contours_for_up:
        area = cv2.contourArea(c_up)
        if area>250:
            x,y,w,h = cv2.boundingRect(c_up)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            if y > prev_y+10:
                cv2.putText(frame, 'Scrolling up', (50, 50), FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
                pyautogui.press('up',presses=2)
            prev_y = y



    cv2.imshow('frame', frame),

    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()