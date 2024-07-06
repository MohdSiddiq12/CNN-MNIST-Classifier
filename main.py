from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

# Load training and eval data

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_eval, y_eval) = mnist.load_data()
x_train, x_eval = x_train / 255.0, x_eval / 255.0 #dividing input by 255 to normalize the inputs,
#usually images range between 0 to 255, 0 being black 255 being white by dividing we scale the values between 0 and 1, 
#so it normalizes the input which is a common practice in neural network!

# Define the model using Keras Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Reshape(target_shape=[28, 28, 1], input_shape=(28, 28)),
    tf.keras.layers.Conv2D(filters=32, kernel_size=[5, 5], padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2),
    tf.keras.layers.Conv2D(filters=64, kernel_size=[5, 5], padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=1024, activation='relu'),
    tf.keras.layers.Dropout(rate=0.4),
    tf.keras.layers.Dense(units=10)
])

# Compile the model
# Compile the model with Stochastic Gradient Descent optimizer,
# Sparse Categorical Crossentropy loss function for integer labels,
# and accuracy metric for evaluation.
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Define training and evaluation data input functions
train_input_fn = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(60000).batch(100)
eval_input_fn = tf.data.Dataset.from_tensor_slices((x_eval, y_eval)).batch(100)

# Train the model
model.fit(train_input_fn, epochs=10)

# Evaluate the model
eval_result = model.evaluate(eval_input_fn)
print(f'\nTest set accuracy: {eval_result[1]}\n')
