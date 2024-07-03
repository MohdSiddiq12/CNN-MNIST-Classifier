from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)(input_layer)

    # Pooling Layer #1
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv1)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)(pool1)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)(conv2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)(pool2_flat)
    dropout = tf.keras.layers.Dropout(rate=0.4)(dense, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.keras.layers.Dense(units=10)(dropout)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
        train_op = optimizer.minimize(loss, tf.compat.v1.train.get_or_create_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    eval_metric_ops = {
        "accuracy": tf.compat.v1.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

feature_columns = [tf.feature_column.numeric_column(key="x", shape=[28, 28])]