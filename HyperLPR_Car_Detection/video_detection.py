import sys
from importlib import reload

reload(sys)
# sys.setdefaultencoding("utf-8")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

fontC = ImageFont.truetype("./Font/platech.ttf", 14, 0)


def drawRectBox(image, rect, addText):
    cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])), (0, 0, 255), 2,
                  cv2.LINE_AA)
    cv2.rectangle(image, (int(rect[0] - 1), int(rect[1]) - 16), (int(rect[0] + 115), int(rect[1])), (0, 0, 255), -1,
                  cv2.LINE_AA)
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    draw.text((int(rect[0] + 1), int(rect[1] - 16)), addText, (255, 255, 255), font=fontC)
    imagex = np.array(img)
    return imagex



'''

'''

def carnum_rec(grr):
    for pstr, confidence, rect in model.SimpleRecognizePlateByE2E(grr):
        if confidence > 0.7:
            image = drawRectBox(grr, rect, pstr + " " + str(round(confidence, 3)))
            print("plate_str:")
            print(pstr)
            print("plate_confidence")
            print(confidence)
            # cv2.namedWindow("enhanced", 0)
            # cv2.resizeWindow("enhanced", 640, 480)
            cv2.imshow("demonstration", image)



def video_capture():
    # 0是代表摄像头编号，只有一个的话默认为0
    capture = cv2.VideoCapture("src/car2.mp4")
    i = 1
    while (True):
        # capture.read()
        # 按帧读取视频，ret, frame是获cap.read()
        # 方法的两个返回值。其中ret是布尔值，如果读取帧
        # 是正确的则返回True，如果文件读取到结尾，它的返回值就为False。frame就是每一帧的图像，是个三维矩阵。
        ref, frame = capture.read()
        if ref:

            # cv2.namedWindow("enhanced", 0)
            # cv2.resizeWindow("enhanced", 640, 480)
            cv2.imshow("demonstration", frame)
            i = i + 1
            if i % 10 == 0:
                i = 0
                #cv2.imshow("num",frame)
                #cv2.waitKey(0)
                carnum_rec(frame)
            # 等待30ms显示图像，若过程中按“Esc”退出
            c = cv2.waitKey(30) & 0xff
            if c == 27:  # ESC 按键 对应键盘值 27
                capture.release()
                break
        else:
            break




import HyperLPR_Car_Detection.HyperLPRLite as pr
import cv2
import numpy as np


if __name__ == '__main__':
    model = pr.LPR("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")
    video_capture()
    cv2.destroyAllWindows()


#
# grr = cv2.imread("src/car.jpg")
# model = pr.LPR("model/cascade.xml", "model/model12.h5", "model/ocr_plate_all_gru.h5")
#




