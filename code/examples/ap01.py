import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-c', '--cpu', action='store_true')
parser.add_argument('-g', '--gpu', action='store_true')
parser.add_argument('-d', '--device', default="gpu")

args = parser.parse_args()
import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np
import random as rn

os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = 'true'
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'

# parameter
epoch= 50
seed = 1234


tf.random.set_seed(seed)
np.random.seed(seed)
rn.seed(seed)

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=32, inter_op_parallelism_threads=32)
#tf.compat.v1.set_random_seed(seed)
#sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

def main():
    #(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar100.load_data()

    train_x, test_x = train_x / 255.0, test_x / 255.0

    model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape = (32, 32, 3)),  # 畳み込み層 #1
            #tf.keras.layers.MaxPooling2D((2, 2)),                                           # プーリング層 #1
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),                           # 畳み込み層 #2
            #tf.keras.layers.MaxPooling2D((2, 2)),                                           # プーリング層 #2
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),  # 畳み込み層 #1
            #tf.keras.layers.MaxPooling2D((2, 2)),                                           # プーリング層 #1
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),                           # 畳み込み層 #2
            tf.keras.layers.MaxPooling2D((2, 2)), 
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(100)
            ],name='example_model_02')

    model.summary()

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    history = model.fit(train_x, train_y, epochs=epoch,
                        validation_data=(test_x, test_y))

    test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)

    print(test_acc)


if args.device.lower() == 'cpu' or args.cpu:
    print('Learning with CPU')
    with tf.device("/cpu:0"):
        main()
elif args.device.lower() == 'gpu' or args.gpu:
    print('Learning with GPU')
    with tf.device("/gpu:0"):
        main()
