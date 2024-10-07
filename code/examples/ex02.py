import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np
import random as rn

os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = 'true'
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'

# パラメータ
epoch= 50
seed = 1234


tf.random.set_seed(seed)
np.random.seed(seed)
rn.seed(seed)

(train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()

train_x, test_x = train_x / 255.0, test_x / 255.0

# TODO 本プログラムを実行して実際の学習プロセスを確認する
model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(3, (3, 3), activation='relu', input_shape = (32, 32, 3)),  # 畳み込み層 #1
        tf.keras.layers.MaxPooling2D((2, 2)),                                             # プーリング層 #1
        tf.keras.layers.Conv2D(3, (3, 3), activation='relu'),                             # 畳み込み層 #2
        tf.keras.layers.MaxPooling2D((2, 2)),                                             # プーリング層 #2
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),                                     # 全結合層 #1
        tf.keras.layers.Dense(10)                                                         # 全結合層 #2 10クラス分類をするので出力は10
        ],name='example_model_02')

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_x, train_y, epochs=epoch,
                    validation_data=(test_x, test_y))

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(frameon=False)
plt.xlabel("epochs")
plt.ylabel("crossentropy")

plt.savefig("loss_"+str(epoch)+"_ex02.png")
plt.show()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend(frameon=False)
plt.xlabel("epochs")
plt.ylabel("Accuracy")

plt.savefig("accu_"+str(epoch)+"_ex02.png")
plt.show()

test_loss, test_acc = model.evaluate(test_x, test_y, verbose=2)

print(test_acc)
