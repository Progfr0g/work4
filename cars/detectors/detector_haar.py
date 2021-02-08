import argparse
from cv2 import dnn_superres
import cv2
import os
from auto_shelves.auto_shelves_processor import AutoShelvesProcessor
import datetime
import time
from PIL import Image
import pytesseract
from pytesseract import Output
import numpy as np
import matplotlib.pyplot as plt
from cars import text_analyze


class Detector_haar():
    def __init__(self):
        # TODO: change to env.~~~~
        self.cascade = '/home/cucumber/somputer-vision/python/cars/haarcascade_russian_plate_number2.xml'
        self.upscale_model_path = "./cars/upscale_model/FSRCNN-small_x3.pb"

        self.sr = dnn_superres.DnnSuperResImpl_create()
        # Read the desired model
        self.sr.readModel(self.upscale_model_path)
        self.sr.setModel("fsrcnn", 3)

        self.loading = 0
        self.limit = 20
        self.series = []

    def rotate_image(self, image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    test = 1

    def clean_by_shape(self, conts, hier, roi_color_up):
        hier = np.squeeze(hier)
        for num, cont in enumerate(conts):
            # if hier[num][2] != -1:
            #     continue
            box = cv2.boxPoints(cv2.minAreaRect(cont))
            box = np.int0(box)
            #
            # cv2.drawContours(roi_color_up, [cont], -1, 110, 2)
            dist_top = np.linalg.norm(box[0] - box[1], 2)
            dist_bottom = np.linalg.norm(box[3] - box[2], 2)

            dist_left = np.linalg.norm(box[0] - box[3], 2)
            dist_right = np.linalg.norm(box[1] - box[2], 2)

            middle_dist_y = int((dist_left + dist_right) * 1.5 / 2)
            middle_dist_x = int((dist_top + dist_bottom) * 1.5 / 2)

            if middle_dist_x == 0 or middle_dist_y == 0 \
                    or float(middle_dist_y / middle_dist_x) <= 1 / 3.0 \
                    or float(middle_dist_x / middle_dist_y) <= 1 / 3.0:

                if hier[num][2] != -1:
                    papa = hier[num][2]
                    mask_2 = np.zeros_like(roi_color_up)
                    cv2.drawContours(mask_2, [cont], 0, 255, -1)
                    for i, elem in enumerate(conts):
                        if hier[i][3] == papa - 1:
                            cv2.drawContours(mask_2, [elem], 0, 0, -1)
                    mask_2 = cv2.dilate(mask_2, (5, 5))
                    roi_color_up = cv2.bitwise_xor(roi_color_up, mask_2)
        return roi_color_up

    def run(self, frame):
        # TODO: check errors
        try:
            plates_text = self.get_number_plate(frame)
            if plates_text != '' and plates_text is not None:
                self.series.append(plates_text)
                if len(self.series) >= self.limit:
                    plates = text_analyze.analyze(self.series)
                    self.series = []
            return plates
        except:
            pass

    def get_number_plate(self, frame):
        # dir = '/home/donteco1/somputer-vision/python/cars/photos/'
        # images = os.listdir(dir)
        # images.sort()
        # print(images)
        # image_path = dir + images[0]
        # img = cv2.imread(image_path)

        img = frame
        # img = cv2.imread('photos/ph1' + '.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        plate_cascade = cv2.CascadeClassifier(self.cascade)
        plates = plate_cascade.detectMultiScale(gray, 1.4, 3, minSize=(10, 20))

        for (x, y, w, h) in plates:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            # Upscale the image
            roi_processed_up = self.sr.upsample(roi_color)
            roi_color_up_end = roi_processed_up.copy()
            roi_processed_up = cv2.cvtColor(roi_processed_up, cv2.COLOR_BGR2GRAY)
            # ================================================================
            # ret, roi_color_up = cv2.threshold(roi_color_up, 127, 255, 0)

            roi_processed_up = cv2.adaptiveThreshold(roi_processed_up, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                     cv2.THRESH_BINARY, 19, 4)
            #
            roi_processed_up = cv2.bitwise_not(roi_processed_up)

            sharpen = np.array((
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]), dtype="int")

            roi_processed_up = cv2.filter2D(roi_processed_up, -1, sharpen)

            roi_processed_up = cv2.morphologyEx(roi_processed_up, cv2.MORPH_OPEN, (10, 10))
            roi_processed_up = cv2.medianBlur(roi_processed_up, 5)
            roi_processed_up = cv2.morphologyEx(roi_processed_up, cv2.MORPH_CLOSE, (8, 8))

            '''1 stage clean by long shape-----------------'''
            conts, hier = cv2.findContours(roi_processed_up, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            roi_processed_up = self.clean_by_shape(conts, hier, roi_processed_up)

            '''2 stage clean for remaining long contours-----------------'''

            conts, hier = cv2.findContours(roi_processed_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for num, cont in enumerate(conts):
                # if hier[num][2] != -1:
                #     continue
                box = cv2.boxPoints(cv2.minAreaRect(cont))
                box = np.int0(box)
                #
                # cv2.drawContours(roi_color_up, [cont], -1, 110, 2)
                dist_top = np.linalg.norm(box[0] - box[1], 2)
                dist_bottom = np.linalg.norm(box[3] - box[2], 2)

                dist_left = np.linalg.norm(box[0] - box[3], 2)
                dist_right = np.linalg.norm(box[1] - box[2], 2)

                middle_dist_y = int((dist_left + dist_right) * 1.5 / 2)
                middle_dist_x = int((dist_top + dist_bottom) * 1.5 / 2)

                if middle_dist_x == 0 or middle_dist_y == 0 \
                        or float(middle_dist_y / middle_dist_x) <= 1 / 3.0 \
                        or float(middle_dist_x / middle_dist_y) <= 1 / 3.0:
                    cv2.drawContours(roi_processed_up, [cont], 0, 0, -1)

            '''3 stage clean by area----------------'''
            conts, hier = cv2.findContours(roi_processed_up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            avg = np.sum([cv2.contourArea(elem) for elem in conts]) / len(conts)
            for cont in conts:

                if cv2.contourArea(cont) < avg * 0.8:
                    cv2.drawContours(roi_processed_up, [cont], -1, 0, -1)

            roi_processed_up = cv2.morphologyEx(roi_processed_up, cv2.MORPH_CLOSE,
                                                cv2.getStructuringElement(cv2.MORPH_RECT, (6, 5)))
            # ================================================================
            # roi_color_up = cv2.resize(roi_color_up, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
            # data = pytesseract.image_to_data(roi_color_up,
            #                                  lang='eng',
            #                                  config='--psm 6 '
            #                                         '--oem 1'
            #                                         '-c tessedit_char_whitelist=0123456789ABCEHKMOPTXYabcehkmoptxy',
            #                                  output_type=Output.DICT)
            #
            # n_boxes = len(data['text'])
            # for i in range(n_boxes):
            #     if int(data['conf'][i]) >= 0:
            #         (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            #         roi_color_up = cv2.rectangle(roi_color_up, (x, y), (x + w, y + h), 120, 1)
            # #

            # frame_sh_0 = int(roi_color_up.shape[0] * 0.1)
            # frame_sh_1 = int(roi_color_up.shape[1] * 0.1)
            #
            # roi_color_up = cv2.copyMakeBorder(roi_color_up, frame_sh_0, frame_sh_0,
            #                                     frame_sh_1, frame_sh_1, cv2.BORDER_CONSTANT, value=0)
            # ================================================================

            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (85, 20))
            roi_rotate = cv2.dilate(roi_processed_up, dilate_kernel)

            conts, hier = cv2.findContours(roi_rotate, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            common_mask = np.zeros_like(roi_rotate)
            for cont in conts:
                mask = np.zeros_like(roi_rotate)
                try:
                    (x, y), (MA, ma), angle = cv2.fitEllipse(cont)
                except:
                    angle = 90
                box = cv2.boxPoints(cv2.minAreaRect(cont))
                box = np.int0(box)
                cv2.drawContours(mask, [box], 0, 255, -1)
                mask = cv2.bitwise_and(roi_processed_up, mask)

                if angle <= 90:
                    mask = self.rotate_image(mask, angle - 90)
                else:
                    mask = self.rotate_image(mask, angle - 90)

                common_mask = cv2.bitwise_or(common_mask, mask)

            roi_rotate = common_mask
            roi_processed_up = cv2.bitwise_not(roi_rotate)

            # plt.subplot(1, 2, 1)
            # plt.imshow(roi_rotate)
            # plt.set_cmap('gray')
            # plt.title('roi_color_up')
            # plt.subplot(1, 2, 2)
            # # plt.imshow(frame)
            # # # plt.show()
            # plt.pause(0.001)

            text = pytesseract.image_to_string(roi_processed_up, lang='eng', config='--psm 6 \
                                                                                 --oem 1 \
                                                            -c tessedit_char_whitelist=0123456789ABCEHKMOPTXYabcehkmoptxy')

            if text != '' and text != None:
                return text.upper()
                # print(text.upper())
                # result.append(text)

# if test == False:
#     # frame = cv2.imread('./photos/GyAQs.png')
#     frame = cv2.imread('./photos/ph4.jpg')
#
#     get_number_plate(frame)
#
# # if test == False:

#     contour = np.array(
#         [[0.525, 0.03333333333333333], [0.98875, 0.041666666666666664], [0.9775, 0.9708333333333333], [0.5375, 0.95]])

# contour = AutoShelvesProcessor.renormalize(frame.shape[0], frame.shape[1], contour)
# prove = [['B666AO750'],
#          ['A888OK55'], ['M001MM39'], ['A082MP97'],
#          ['O555OO123'], ['A517MP97'],['A186MP97'], ['B776YC77'],
#          ['A652MP97'],
#          ['A001AA40'],
#          ['H001AX777'],
#          ['M555TP55'],
#          ['E027KX94'], ['A116YP116'], ['H001AX777']]
