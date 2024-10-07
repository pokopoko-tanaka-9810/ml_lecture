import numpy as np
import datetime
import cv2
import keras.utils as image

preds_list = ['paper', 'rock', 'scissors'] # abc順にディレクトリを読み込んだ関係で推測ラベルの順番は左の通り
metric = 'val_loss'
checkpoint_mode = 'min'
model_base_path = './model/'
model_name_prefix = 'task02_'

def capture_device(model, capture, dev, model_type, input_size, decode_predictions = None):
    while True:
        # cameraデバイスから画像をキャプチャ
        ret, frame = capture.read()
        if not ret:
            k = ord('+')
            return k

        # DenseNet121画像判定
        resize_frame = cv2.resize(frame, (300, 224))            # 640x480(4:3) -> 300x224(4:3)に画像リサイズ
        trim_x, trim_y = int((300-224)/2), 0                    # 判定用に224x224へトリミング
        trim_h, trim_w = input_size, input_size
        trim_frame = resize_frame[trim_y : (trim_y + trim_h), trim_x : (trim_x + trim_w)]
        x = image.img_to_array(trim_frame)
        x = np.expand_dims(x, axis=0)/255.0                     # 正規化

        preds = model.predict(x, verbose=0)                                # 画像AI判定
        disp_frame = frame
        txt1 = "model is Original net;q"
        txt2 = "camera device No.(" + str(dev) + ")"
        txt3 = "[+] : Change Device"
        txt4 = "[s] : Image Capture"
        txt5 = "[ESC] or [q] : Exit"

        cv2.putText(disp_frame, txt1, (10,  30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(disp_frame, txt2, (10,  60), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(disp_frame, txt3, (10,  90), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(disp_frame, txt4, (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(disp_frame, txt5, (10, 150), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # 画像判定文字出力
        if model_type == 'transfer':
            pred_arg = preds[0].argmax()

            result = 'Prediction Result - {0}:{1:.2f}%'.format(preds_list[pred_arg], preds[0][pred_arg])
            #output1 = 'rock:{0:.2f}%'.format(preds[0][1])
            #output2 = 'scissors:{0:.2f}%'.format(preds[0][2])
            #output3 = 'paper:{0:.2f}%'.format(preds[0][0])

            cv2.putText(disp_frame, result, (10, 300), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
            #cv2.putText(disp_frame, output1, (10, 330), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
            #cv2.putText(disp_frame, output3, (10, 360), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
            #cv2.putText(disp_frame, output3, (10, 360), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)

        elif model_type == 'original':
            output1 = 'No.1:{0}:{1}%'.format(decode_predictions(preds, top=3)[0][0][1], int(decode_predictions(preds, top=3)[0][0][2] * 100))
            output2 = 'No.2:{0}:{1}%'.format(decode_predictions(preds, top=3)[0][1][1], int(decode_predictions(preds, top=3)[0][1][2] * 100))
            output3 = 'No.3:{0}:{1}%'.format(decode_predictions(preds, top=3)[0][2][1], int(decode_predictions(preds, top=3)[0][2][2] * 100))

            cv2.putText(disp_frame, output1, (10, 300), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(disp_frame, output2, (10, 330), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(disp_frame, output3, (10, 360), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)

        # カメラ画面出力
        cv2.imshow('camera', disp_frame)

        # 1msec待ってキー取得
        k = cv2.waitKey(1) & 0xFF

        # [ESC] or [q]を押されるまで画面表示し続ける
        if (k == ord('q')) or (k == 27):
            return k

        # [+]でdevice変更
        if k == ord('+'):
            txt = "Change Device. Please wait... "
            XX = int(disp_frame.shape[1] / 4)
            YY = int(disp_frame.shape[0] / 2)
            cv2.putText(disp_frame, txt, (XX, YY), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('camera', disp_frame)
            cv2.waitKey(1) & 0xFF
            return k

        # [s]で画面に表示された画像保存
        elif k == ord('s'):
            cv2.imwrite('camera_dsp{}.{}'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'), "png"), disp_frame)



def camera(model, dev, model_type, input_size, decode_predictions):

    while True:
        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        ret = capture_device(model, capture, dev, model_type, input_size, decode_predictions)

        if (ret == ord('q')) or (ret == 27):
            # リソース解放
            capture.release()
            cv2.destroyAllWindows()
            break

        if ret == ord('+'):
            dev += 1

            if dev == 9:
                dev = 0
