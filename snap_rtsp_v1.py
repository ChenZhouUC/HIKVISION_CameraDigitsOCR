import cv2
import pytesseract
from PIL import Image
import numpy as np
from copy import deepcopy

MAX_SUB = 2
MIN_SUB = 0


def adathre(crop_num, crit=(225, 30), larger=3, thre=0.1):
    gray = cv2.cvtColor(crop_num, cv2.COLOR_BGR2GRAY)
    thresh_1, thresh_2 = gray.copy(), gray.copy()
    cv2.threshold(gray, crit[0], 255, cv2.THRESH_BINARY_INV, thresh_1)
    cv2.threshold(gray, crit[1], 255, cv2.THRESH_BINARY, thresh_2)
    if np.mean(255-thresh_1)/255 > thre:
        thresh_1 = (np.ones(thresh_1.shape)*255).astype(np.uint8)
    if np.mean(255-thresh_2)/255 > thre:
        thresh_2 = (np.ones(thresh_2.shape)*255).astype(np.uint8)
    thresh = cv2.bitwise_and(thresh_1, thresh_2)
    thresh = cv2.resize(
        thresh, (int(thresh.shape[1]*larger), int(thresh.shape[0]*larger)))
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # thresh = cv2.dilate(thresh, kernel)
    return thresh


def recog_num(pil_img):
    num = pytesseract.image_to_string(pil_img, lang='osd', config='--psm 6 --oem 2 digit_only')
    return num


def increase_check(total, last_total):
    sub = np.array(total) - np.array(last_total)
    if ((sub <= MAX_SUB).sum() == len(sub)) and ((sub >= MIN_SUB).sum() == len(sub)):
        return True
    else:
        return False


if __name__ == '__main__':
    test = Image.open("test.png")
    text = recog_num(test)
    print(text)
    input()
    cap = cv2.VideoCapture(
        "rtsp://admin:pass1234@192.168.192.12:554/h264/ch39/sub/av_stream")
    counter = 0
    total = []
    while(1):
        ret, frame = cap.read()
        if ret:
            try:
                if counter == 0:
                    last_total = deepcopy(total)
                    total = frame[:66, 518:580].copy()
                    total = adathre(total)
                    cv2.imshow('total', total)
                    total = recog_num(Image.fromarray(total)).replace(
                        " ", "").replace("\t", "").split('\n')
                    print(total)
                    total = [int(i) for i in total]

                    if increase_check(total, last_total):
                        print("[enter:", total[0], "][leave:",
                              total[1], "][dupli:", total[2], "]")
                        if (np.array(total) >= 10).sum() >= 1:
                            print(total, last_total)
                            cv2.imshow('total', total)
                            cv2.waitKey(-1)
                            input()

                counter += 1

                if counter >= 25:
                    counter = 0

                frame = cv2.rectangle(frame, (518, 0), (528, 20),
                                      (0, 255, 0, 255), 1, 1)
                frame = cv2.rectangle(frame, (518, 23), (528, 43),
                                      (255, 0, 0, 255), 1, 1)
                frame = cv2.rectangle(frame, (518, 46), (528, 66),
                                      (0, 0, 255, 255), 1, 1)

                cv2.imshow('rtsp', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    frame = cv2.resize(frame, (320, 240))
                    cv2.imwrite('bg.png', frame)
                    break

            except Exception as e:
                print(e)
                continue
        else:
            print("Camera Disconnected! Try Reconnecting...")
            continue
