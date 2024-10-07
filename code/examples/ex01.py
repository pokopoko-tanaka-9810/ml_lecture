import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import matplotlib.pyplot as plt

# パラメータ
epoch= 10

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()

train_x, test_x = train_x / 255.0, test_x / 255.0

# TODO モデルの構成を確認する
model = tf.keras.models.Sequential([
        # 本日の講義で取りあげた部分 ↓
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=(32, 32, 3)),  # 畳み込み層 #1
        tf.keras.layers.MaxPooling2D((2, 2)),                                           # プーリング層 #1
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),                           # 畳み込み層 #2
        tf.keras.layers.MaxPooling2D((2, 2)),                                           # プーリング層 #2
        # 本日の講義で取りあげた部分 ↑
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10)
        ],name='example_model_01')


model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_x, train_y, epochs=epoch,
                    validation_data=(test_x, test_y))

test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)

print(test_acc)
