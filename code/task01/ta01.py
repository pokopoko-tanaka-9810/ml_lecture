import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import config
import datetime as dt

os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = 'true'
os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'


session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=32, inter_op_parallelism_threads=32)
tf.compat.v1.set_random_seed(config.seed)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

dt_now = dt.datetime.now()
time_stamp = str(dt_now.year).zfill(4) + str(dt_now.month).zfill(2) + str(dt_now.day).zfill(2) + str(dt_now.hour).zfill(2) + str(dt_now.minute).zfill(2) + '.h5'
current_model_path = config.model_path + time_stamp
# ここから上は変えないでください


# パラメータ
epoch = 5
batch_size = 1024
learning_rate = 0.001


# モデルを構成する関数
def build_model(class_num, in_size_x, in_size_y, in_dim):
    input_shape = (in_size_x, in_size_y, in_dim)
    input_w = tf.keras.layers.Input(shape=input_shape)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(4, (3, 3), kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.BatchNormalization()) # バッチ正規化 : 過学習を抑えるための手法
    model.add(tf.keras.layers.Activation('selu'))
    model.add(tf.keras.layers.AveragePooling2D((2, 2), strides=2))
    model.add(tf.keras.layers.Conv2D(4, (3, 3), kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('selu'))
    model.add(tf.keras.layers.AveragePooling2D((2, 2), strides=2))
    model.add(tf.keras.layers.Conv2D(4, (3, 3), kernel_initializer='he_uniform', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('selu'))
    model.add(tf.keras.layers.AveragePooling2D((2, 2), strides=2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(16, kernel_initializer='he_uniform', activation='selu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(16, kernel_initializer='he_uniform', activation='selu'))

    model.add(tf.keras.layers.Dense(class_num, activation='softmax'))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# モデルを学習させる関数
def train(X_train_data, y_train_data):   

    in_size_x = X_train_data.shape[1]
    in_size_y = X_train_data.shape[2]
    in_dim = X_train_data.shape[3]
    nw = build_model(config.class_num, in_size_x, in_size_y, in_dim)

    # generating datasets for training and validation
    # shuffle → Trueであればランダムに分ける. Falseであれば先頭からx番目までをtrain, x+1番目以降をvalとする.
    # random_state → shuffleの時の乱数を指定. 再現性を持たせるために今回は常に同じ値にしている.
    # train_size → 0~1の実数で指定した場合はtrainとvalの比率, 0以上の整数で指定した場合はそのデータセットを構成するデータの数となる.
    X_train, X_val, y_train, y_val = train_test_split(X_train_data, y_train_data,
                                                      shuffle = True,
                                                      random_state = config.seed,
                                                      train_size = config.train_size)

    # saving model
    # 学習中にmonitorで指定した値がmodeで指定した順列で更新した際にそのモデルを一時的に保存する.
    # monitorをval_loss, modeをminとした場合, 学習中の検証でval_lossがこれまでで最も小さい値(=正解との誤差が最も小さい)になったときに
    # そのモデルを一時保存しておき, すべての学習が終わったタイミングで一時保存されていたモデルを出力する.
    # これにより, 過学習してしまった場合でも学習の中で一番性能のよいモデルを保存することが可能となる.
    modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath = current_model_path,
                                                         monitor=config.metric,
                                                         verbose=0,
                                                         save_best_only=True,
                                                         save_weights_only=False,
                                                         mode=config.checkpoint_mode)

    # Running learning
    history = nw.fit(X_train, y_train,
                     validation_data=(X_val, y_val),
                     epochs=epoch,
                     batch_size=batch_size,
                     callbacks=[modelCheckpoint],
                     verbose=1)

    return history


if __name__ == '__main__':
    # データセットの内容を確認
    # show_datasets()

    # 学習
    X_train_data, y_train_data, X_test, y_test = config.data_loader()
    history = train(X_train_data, y_train_data)
    config.draw_learning_curve(history, time_stamp, epoch)
    config.test(current_model_path, X_test, y_test)
