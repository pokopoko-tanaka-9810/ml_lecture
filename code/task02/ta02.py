import os
import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

from config import config
from config import gen_dataset as gd

def main(func_mode):
    input_size = 224
    img_size = input_size

    # 実習2
    # ↓TODO↓
    img_noise = 0
    img_rotate = 0
    img_flip = 0
    # ↑TODO↑

    if func_mode == 'check':
        gd.gen_data(mode = func_mode,
                    img_noise = img_noise,
                    img_size = img_size,
                    img_rotate = img_rotate,
                    img_flip = img_flip)

    elif func_mode == 'transfer':
        model_type = func_mode
        decode_predictions = None
        # 学習用データの生成
        print('Do you generate new dataset? Existing dataset will be overwrited. Y/n')
        gd_in = input()
        if gd_in.upper() == 'Y' or gd_in.upper() == 'YES':
            print('Now generating...')
            gd.gen_data(img_noise = img_noise,
                        img_size = img_size,
                        img_rotate = img_rotate,
                        img_flip = img_flip)
            print('Data Generation Done.')

        # 学習/検証/テスト用データの読み込み
        data_list = ['/train', '/valid', '/test']
        X_train, y_train = gd.data_load(data_list[0])
        X_valid, y_valid = gd.data_load(data_list[1])
        #X_test, y_test = gd.data_load(data_list[2])

        # 転移学習
        input_shape = (input_size, input_size, 3)

        # 確認ポイント
        # include_topをFalseにしてモデルをロードしている
        # →最後の全結合層(=top)は除いた部分のみを使用するため
        base_model = tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape = input_shape)

        # 実習1
        transfer_model = tf.keras.models.Sequential()
        # 前段は全結合層をそのまま使用する
        transfer_model.add(base_model) 
        # 後段の全結合層を新たに定義
        # ↓TODO↓
        # transfer_model.add(XXX)
        # transfer_model.add(XXX)
        # ...       
        # ↑TODO↑

        transfer_model.summary()

        # 学習済みモデルの重みを固定(更新しない)
        base_model.trainable = False

        transfer_model.summary()

        # 転移学習での学習率は低め
        learning_rate = 1e-6
        transfer_model.compile(loss='categorical_crossentropy',
                               optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                               metrics=['accuracy'])

        date_data_now = datetime.datetime.now()
        str_now = str(date_data_now.year) + str(date_data_now.month).zfill(2) + str(date_data_now.day).zfill(2) + str(date_data_now.hour).zfill(2) + str(date_data_now.minute).zfill(2)
        current_model_path = config.model_base_path + config.model_name_prefix + str_now + ".keras"

        # 学習したモデルを保存
        modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(filepath = current_model_path,
                                            monitor=config.metric,
                                            verbose=0,
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode=config.checkpoint_mode)


        transfer_model.fit(X_train, y_train,
                           epochs=20,
                           batch_size=32,
                           callbacks=[modelCheckpoint],
                           validation_data = (X_valid, y_valid))

        load_model = tf.keras.models.load_model(current_model_path)

    elif func_mode == 'original':
        model_type = func_mode
        load_model = tf.keras.applications.densenet.DenseNet121(weights='imagenet')
        decode_predictions = tf.keras.applications.densenet.decode_predictions

    elif func_mode == 'load':
        model_type = 'transfer'
        decode_predictions = None

        model_path = input("Enter Model Name:")
        if not model_path[-6:] == '.keras':
            model_path += '.keras'
        load_model = tf.keras.models.load_model(config.model_base_path + model_path)


    # カメラで識別を行う
    config.camera(load_model,
                  dev=0,
                  model_type = model_type,
                  input_size = input_size,
                  decode_predictions = decode_predictions)

if __name__ == '__main__':
    while(True):
        input_type = input('Choose Function Mode - Enter 1, 2 3 or 4.\n \
1:Image Detection Using Original Model\n \
2:Transfer Learning for recognition Janken Hand\n \
3:Load Learned Model\n \
4:Check Dataset Image\n \
Mode:')
        if input_type == '1':
            func_mode = 'original'
            break
        elif input_type == '2':
            func_mode = 'transfer'
            break     
        elif input_type == '3':
            func_mode = 'load'
            break 
        elif input_type == '4':
            func_mode = 'check'
            break   
        else:
            print('Please Enter either 1, 2, 3 or 4.')

    main(func_mode)
