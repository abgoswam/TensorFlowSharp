# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 18:49:30 2018

@author: agoswami
"""

import argparse
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt

def load_graph(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="models/frozen_saved_model.pb", type=str, help="Frozen model file to import")
    args = parser.parse_args()

    graph = load_graph(args.frozen_model_filename)

    # We can list operations
    for op in graph.get_operations():
        print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions
    x = graph.get_tensor_by_name('prefix/I:0')
    y = graph.get_tensor_by_name('prefix/O:0')

    with tf.Session(graph=graph) as sess:
        y_out = sess.run(y, feed_dict={
            x: [[1, 2, 2],[4, 5, 5]] # < 45
        })
        print(y_out) # [[ False ]] Yay!