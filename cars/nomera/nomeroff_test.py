# Import all necessary libraries.
import os
import time
import numpy as np

from matplotlib import pyplot as plt
import cv2
import sys
import matplotlib.image as mpimg
from python.cars.nom_lib import filters, RectDetector, TextDetector, OptionsDetector, Detector, textPostprocessing, \
    textPostprocessingAsync

# change this property
NOMEROFF_NET_DIR = os.path.abspath('')

# specify the path to Mask_RCNN if you placed it outside Nomeroff-net project
MASK_RCNN_DIR = os.path.join(NOMEROFF_NET_DIR, 'Mask_RCNN')
MASK_RCNN_LOG_DIR = os.path.join(NOMEROFF_NET_DIR, 'logs')

sys.path.append(NOMEROFF_NET_DIR)
# Import license plate recognition tools.

# Initialize npdetector with default configuration file.
nnet = Detector(MASK_RCNN_DIR, MASK_RCNN_LOG_DIR)
# nnet.loadModel(NOMEROFF_NET_DIR+'/Mask_RCNN/'+'mask_rcnn_numberplate_0640_2019_06_24.pb')

rectDetector = RectDetector()

optionsDetector = OptionsDetector()
optionsDetector.load(NOMEROFF_NET_DIR + '/Mask_RCNN/' + 'numberplate_options_2019_06_27.h5')

# Initialize text detector.
textDetector = TextDetector.get_static_module("ru")()
textDetector.load(NOMEROFF_NET_DIR + '/Mask_RCNN/' + 'anpr_ocr_ru_6-cpu.h5')

# Detect numberplate
current_time = time.time()
timer = current_time
while True:
    current_time = time.time()
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('rtsp://admin:oYk3JdUS@192.168.1.107:554/live/main')
    ret, frame = cap.read()
    # frame = np.array(frame)
    # frame = cv2.resize(frame, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR, dsize=None)
    # img = mpimg.imread(img_path)
    if ret:
        if current_time >= timer:
            NP = nnet.detect_serving([frame])
            # Generate image mask.
            cv_img_masks = filters.cv_img_mask(NP)

            # Detect points.
            arrPoints = rectDetector.detect(cv_img_masks)
            zones = rectDetector.get_cv_zonesBGR(frame, arrPoints)

            # find standart
            regionIds, stateIds, countLines = optionsDetector.predict(zones)
            regionNames = optionsDetector.getRegionLabels(regionIds)
            # print(regionIds)
            # print(countLines)
            # print(regionNames)

            if countLines != [0, 0]:
                # for cont in countLines:
                #     cv2.drawContours(frame, cont, 0, (100, 250, 0), 1)
                # # find text with postprocessing by standart
                textArr = textDetector.predict(zones)
                # for num, zone in enumerate(zones):
                #     cv2.imwrite('zone_' + str(num) +'.jpg', zone)
                textArr = textPostprocessing(textArr, regionNames)
                # for num, text in enumerate(textArr):
                #     cv2.putText(frame, text, (10, 10 + 15*num), cv2.FONT_HERSHEY_DUPLEX, 1, (40*num, 100+40*num, 0), 2)
                print(textArr)
            timer = current_time + 6
        plt.imshow(frame)
        plt.pause(0.2)
    # ['JJF509', 'RP70012']
