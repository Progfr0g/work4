# Import all necessary libraries.
import os
import cv2

from matplotlib import pyplot as plt
import sys
from python.cars.nom_lib import filters, RectDetector, TextDetector, OptionsDetector, Detector, textPostprocessing, \
    textPostprocessingAsync


class Detector_GPU():
    def __init__(self):
        # change this property
        self.NOMEROFF_NET_DIR = '/home/cucumber/somputer-vision/python/cars/nomera'

        # specify the path to Mask_RCNN if you placed it outside Nomeroff-net project
        self.MASK_RCNN_DIR = os.path.join(self.NOMEROFF_NET_DIR, 'Mask_RCNN')
        self.MASK_RCNN_LOG_DIR = os.path.join(self.NOMEROFF_NET_DIR, 'logs')

        sys.path.append(self.NOMEROFF_NET_DIR)
        # Import license plate recognition tools.

        # Initialize npdetector with default configuration file.
        self.nnet = Detector(self.MASK_RCNN_DIR, self.MASK_RCNN_LOG_DIR)
        # nnet.loadModel(NOMEROFF_NET_DIR+'/Mask_RCNN/'+'mask_rcnn_numberplate_0640_2019_06_24.pb')

        self.rectDetector = RectDetector()

        self.optionsDetector = OptionsDetector()
        self.optionsDetector.load(self.NOMEROFF_NET_DIR + '/Mask_RCNN/' + 'numberplate_options_2019_06_27.h5')

        # Initialize text detector.
        self.textDetector = TextDetector.get_static_module("ru")()

        self.textDetector.load(self.NOMEROFF_NET_DIR + '/Mask_RCNN/' + 'anpr_ocr_ru_6-cpu.h5')

        # Detect numberplate

    def run(self, frame):
        NP = self.nnet.detect_serving([frame])
        # Generate image mask.
        cv_img_masks = filters.cv_img_mask(NP)

        # Detect points.
        arrPoints = self.rectDetector.detect(cv_img_masks)
        zones = self.rectDetector.get_cv_zonesBGR(frame, arrPoints)

        # find standart
        regionIds, stateIds, countLines = self.optionsDetector.predict(zones)
        regionNames = self.optionsDetector.getRegionLabels(regionIds)
        # print(regionIds)
        # print(countLines)
        # print(regionNames)

        if countLines != [0, 0]:
            # for cont in countLines:
            #     cv2.drawContours(frame, cont, 0, (100, 250, 0), 1)
            # # find text with postprocessing by standart
            textArr = self.textDetector.predict(zones)
            # for num, zone in enumerate(zones):
            #     cv2.imwrite('zone_' + str(num) +'.jpg', zone)
            textArr = textPostprocessing(textArr, regionNames)
            for num, text in enumerate(textArr):
                cv2.putText(frame, text, (10, 10 + 15 * num), cv2.FONT_HERSHEY_DUPLEX, 1, (40 * num, 100 + 40 * num, 0),
                            2)

        plt.imshow(frame)
        plt.pause(0.2)
