import cv2
import numpy as np
from PIL import Image
import time
import os
from copy import deepcopy

GAP_W = 8
GAP_H = 24
W = 10
H = 13
START_W = 518
START_H = 0

MAX_POS_W = 5
MAX_POS_H = 3

MAX_GAP = 10
MIN_GAP = 0


def get_rect(pos_x, pos_y):
    l = START_W + pos_x*GAP_W
    r = START_W + pos_x*GAP_W + W
    t = START_H + pos_y*GAP_H
    d = START_H + pos_y*GAP_H + H
    return l, t, r, d


def convert_std(img_mat, top=1, degree=2):
    if len(img_mat.shape) >= 3:
        img_mat = cv2.cvtColor(img_mat, cv2.COLOR_BGR2GRAY)
    digit_map = cv2.threshold(
        img_mat, 0, top, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]
    if (digit_map[0, :].sum() + digit_map[-1, :].sum() + digit_map[:, 0].sum()
            + digit_map[:, -1].sum())/(W + H - 2)/2 >= top/2:
        digit_map = cv2.threshold(
            img_mat, 0, top, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)[1]
    digit_map = digit_map.astype('float')
    digit_map = digit_map/np.linalg.norm(digit_map, degree)
    return digit_map


def load_digits(num, ext='.png', dir_path='./digits', top=1, degree=2):
    file_path = os.path.join(dir_path, str(num)+ext)
    digit = cv2.imread(file_path, 0)
    digit_map = convert_std(digit, top, degree)
    return digit_map


def matrix_similarity(mat_A, mat_B):
    return (mat_A*mat_B).sum()


def decide_num(cand, digits, crit=0.8):
    score = 0
    idx = 0
    for i, d in enumerate(digits):
        temp_score = matrix_similarity(d, cand)
        if temp_score >= score:
            idx = i
            score = temp_score
        print(temp_score)
    if score > crit:
        return idx
    else:
        return -1


def increase_check(total, last_total):
    sub = np.array(total) - np.array(last_total)
    if ((sub <= MAX_GAP).sum() == len(sub)) and ((sub >= MIN_GAP).sum() == len(sub)):
        return True
    else:
        return False


if __name__ == '__main__':

    digits = []
    for i in range(10):
        digits.append(load_digits(i))
        # cv2.imshow(str(i), digits[-1])
        # print(digits[-1])
    decide_num(digits[7], digits)
    input()
    cap = cv2.VideoCapture(
        "rtsp://admin:pass1234@192.168.192.12:554/h264/ch39/sub/av_stream")
    counter = 0
    count_list_last = []
    while(True):
        ret, frame = cap.read()
        if ret:
            try:
                # position = [1, 1, 11]
                # l, t, r, d = get_rect(position[0], position[1])
                # crop = frame[(t+1):d, (l+1):r]
                # cv2.imwrite(str(position[2])+'.png', crop)
                # frame = cv2.rectangle(frame, (l, t), (r, d),
                #                       (0, 255, 0, 255), 1, 1)
                if counter == 0:

                    count_list = []
                    for pos_h in range(MAX_POS_H):
                        temp_num = ''
                        for pos_w in range(MAX_POS_W):
                            l, t, r, d = get_rect(pos_w, pos_h)
                            crop = frame[(t+1):d, (l+1):r]
                            temp_mat = convert_std(crop)
                            temp_digit = decide_num(temp_mat, digits)
                            if temp_digit >= 0:
                                temp_num += str(temp_digit)
                            else:
                                break
                        count_list.append(int(temp_num))

                    if len(count_list_last) > 0 and increase_check(count_list, count_list_last):
                        print(count_list)
                        count_list_last = deepcopy(count_list)
                    elif len(count_list_last) == 0:
                        print(count_list)
                        count_list_last = deepcopy(count_list)

                counter += 1
                if counter == 25:
                    counter = 0

                cv2.imshow('rtsp', frame)
                cv2.waitKey(30)

            except Exception as e:
                print(e)
                continue
        else:
            print("Camera Disconnected! Try Reconnecting...")
            time.sleep(1)
            continue
