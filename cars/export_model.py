import numpy as np
import os
import os.path
import tensorflow.compat.v1 as tf

PATH_TO_CKPT = os.path.abspath(
    '/home/cucumber/somputer-vision/python/core/models/ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb')

export_dir = os.path.join('/home/cucumber/export_cpu')

# INPUT_NODES = ('', '') # 3 elem

OUTPUT_NODES = ('image_tensor:0', 'detection_boxes:0', 'detection_scores:0', 'detection_classes:0')

graph_def = tf.GraphDef()
with tf.gfile.GFile(PATH_TO_CKPT, "rb") as f:
    graph_def.ParseFromString(f.read())

graph = tf.Graph()

with graph.as_default():
    input_image, detection_boxes, detection_scores, detection_classes = tf.import_graph_def(
        graph_def, return_elements=OUTPUT_NODES)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(graph=graph, config=sess_config)

    # Create SavedModelBuilder class
    # defines where the model will be exported
    export_path_base = "/home/cucumber/cars/export"
    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(0)))
    print('Exporting trained model to', export_path)

    current_graph = tf.get_default_graph()

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    input_image_tensor = tf.saved_model.build_tensor_info(graph.get_tensor_by_name('import/image_tensor:0'))

    output_detection_boxes = tf.saved_model.build_tensor_info(graph.get_tensor_by_name("import/detection_boxes:0"))
    output_detection_scores = tf.saved_model.build_tensor_info(graph.get_tensor_by_name("import/detection_scores:0"))
    output_detection_classes = tf.saved_model.build_tensor_info(graph.get_tensor_by_name("import/detection_classes:0"))

    prediction_signature = (
        tf.saved_model.build_signature_def(
            inputs={'image_tensor:0': input_image_tensor
                    },

            outputs={"detection_boxes:0": output_detection_boxes,
                     "detection_scores:0": output_detection_scores,
                     "detection_classes:0": output_detection_classes
                     },

            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'serving_default':
                prediction_signature,
        })

    builder.save(as_text=False)
