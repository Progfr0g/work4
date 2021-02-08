import time
# from queues.queue_processor import QueueProcessor
from core.logs import Logs, TAG_QUEUE, TAG_REPORTS, TAG_CUSTOM_STREAM
from core.fps_counter import FpsCounter
from core.report import Report
from cars import cars_processor
from core.fpm_adaptation import FpmAdaptation
# from queues.queue import Queue

import env


def startModule(core_factory, module_props, frames_callback, exit_timeout, model):
    """
                            Запуск модуля очередей
        @param core_factory: фабрика core объектов (см. core_factory.factory)
        @param module_props: свойства модуля (см core_factory.module_props)
        @type frames_callback: function
        @param frames_callback: колбэк для отправки кадров наружу из модуля
        @type exit_timeout: int
        @param exit_timeout: timeout in seconds when module will should stop
        @param model: объект управления данными очередей (см. queues.queues_model)
    """

    camera_id = module_props.target_camera_id
    module_name = module_props.module_name

    configs = core_factory.configs
    requests_http = core_factory.requests_http
    database = core_factory.database
    frames_provider = core_factory.frames_provider_rgb

    sockets = core_factory.sockets

    frames_provider.start()

    processor = cars_processor.CarsProcessor(configs)

    # cv algorithms init
    # TODO : cars processor
    #  processor = carsProcessor(configs, requests_http)
    #  queues = model.loadQueues(camera_id, processor, requests_http, configs, sockets)
    #  report.set(REPORT_QUEUES_COUNT, len(queues))

    fps_counter = FpsCounter(module_name, camera_id, database, env.FRAMES_COUNT_INTERVAL)

    delay = 0.9
    timer = time.time()
    while True:
        current_time = time.time()
        frame_rgb = frames_provider.getNextFrame()

        fps_counter.inc()

        if current_time >= timer:
            detections = processor.detect(frame_rgb)
            timer = current_time + delay

        # for element in detections:
