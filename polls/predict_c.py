import tensorflow as tf
import sys
import requests
import os
# change this as you see fit
def breeds(image_path):
    
    response = requests.get(image_path)
    img_data = response.content
    filename = os.path.basename(image_path)
    with open(filename, 'wb') as f:
        f.write(img_data)
    img_file = image_path[40:]
    
    # Read in the image_data
    image_data = tf.gfile.GFile(img_file, 'rb').read()
    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line in tf.gfile.GFile("polls/nn_files/retrained_labels.txt")]

    # Unpersists graph from file
    with tf.gfile.GFile("polls/nn_files/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:

	# Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        # Sort to show labels of first prediction in order of confidence

        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        cnt = 0
        for node_id in top_k:
            human_string = label_lines[node_id]
            if cnt == 0 :
                result = human_string
            score = predictions[0][node_id]
            cnt = cnt + 1
    return result
