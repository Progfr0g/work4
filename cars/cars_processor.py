from cars.detectors import detector_haar, detector_gpu  # , detector_cpu, detector_gpu


class CarsProcessor():
    def __init__(self, configs):
        self.MODEL = 1
        if configs.get("use_haar"):
            self.detector = detector_haar.Detector_haar()
        # elif configs.get("use_cpu"):
        #     self.detector = detector_cpu.Detector_cpu()
        elif configs.get("use_gpu"):
            self.detector = detector_gpu.Detector_GPU()

        # self.detector = detector_haar.Detector_haar()

    def detect(self, frame):
        number_plates = self.detector.run(frame)
        if number_plates is not None:
            print(number_plates)
