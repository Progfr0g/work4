import tensorflow as tf

input_model_filepath = './saved_model.pb'

with tf.gfile.GFile(input_model_filepath, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# import graph_def
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def)

# print operations
for op in graph.get_operations():
    if op.type == "Placeholder":
        print(op.name)
