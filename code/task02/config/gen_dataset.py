import os
import numpy as np
import cv2
import random
import urllib.request
import shutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

label_list = ['paper', 'rock', 'scissors']
data_list = ['/train', '/valid', '/test']
data_path = './data'
data_file_name = ''
url= "https://public.roboflow.com/ds/hfUFTOf07d?key=ZD4TxpUszt"

download_path ='./data.zip'


def extract_dataset(url, download_path):
    urllib.request.urlretrieve(url, download_path)
    shutil.unpack_archive(download_path, data_path)


def clear_data():
    for l in data_list:
        if os.path.exists(data_path + data_file_name + l + l + '_data.npz'):
            os.remove(data_path + data_file_name + l + l + '_data.npz')
    return 0

def rotate_random(img):
    height, width = img.shape[:2]
    center = (width//2, height//2)
    random_angle = random.randrange(360)
    trans = cv2.getRotationMatrix2D(center, random_angle, 1.0)
    image2 = cv2.warpAffine(img, trans, (width,height), borderValue=(255, 255, 255))

    return image2

def make_data(mode, folder_name, img_up_num = 1, img_noise = 0 , img_size = 0, clip_size = 0 ,img_rotate = 0, img_flip = 0, split_num = 0, shuffle_flag = True):
    # img_up_num: 2以上で同じ画像を複製してデータセットを増やす. (指定した値)倍に画像が増える. 増やした各データに対してノイズ付加や回転等の処理を行う場合は異なる処理内容となる.
    # img_flip: 1で画像を左右反転させる.
    # img_rotate: 1で画像をランダムに回転させる. 回転でできた空白部分はすべて白で塗りつぶされる.
    # clip_size:  1以上の値で画像をランダムにクリッピングする. 現在の実装では手や指の一部がはみ出る可能性アリのため使っていない.
    # img_size: 1以上の値で, 1辺が指定した値の正方形画像のサイズに変更する.
    # img_noise: 1で画像にランダムなノイズを加える
    # split_num: 1以上の値で先頭から指定した分だけデータのデータを出力する. ランダムか先頭かからは下のshuffle_flagで決める
    # shuffle_flag: Trueで出力するデータセットをシャッフルする.Falseだと読み込んだディレクトリの順番(アルファベット順)

    train_image = []
    train_label = []

    cnt = 0
    # フォルダ内のディレクトリの読み込み
    classes = os.listdir(folder_name)
    for i, d in enumerate(classes):
        files = os.listdir( folder_name + '/' + d  )

        tmp_image = []
        tmp_label = []
        for f in files:
            # 1枚の画像に対する処理
            if not 'jpg' in f:# jpg以外のファイルは無視
                continue

            # 画像読み込み
            img = cv2.imread(folder_name+ '/' + d + '/' + f)
            # one_hot_vectorを作りラベルとして追加
            label = np.zeros(len(classes))
            label[i] = 1
            for _ in range(img_up_num):
                tmp_img = img
                # 画像処理

                if img_flip != 0:
                    if np.random.choice([0, 1], 1)[0] == 0:
                        tmp_img = cv2.flip(tmp_img, img_flip)

                if  img_rotate != 0:
                    tmp_img = rotate_random(tmp_img)

                #if clip_size != 0:
                #    tmp_img = random_clip(tmp_img, clip_size)

                if img_size != 0:
                    tmp_img = cv2.resize(tmp_img, (img_size, img_size))

                if img_noise != 0:
                    mu = 0
                    sigma = 5
                    noise = np.random.normal(loc = mu, scale = sigma, size = (img_size, img_size, 3))
                    min_noise = noise.min()
                    max_noise = noise.max()
                    p = min_noise/(max_noise - min_noise)
                    tmp_img = tmp_img + 3*(noise+p-0.5)
                    tmp_img = np.where(tmp_img > 255, 255, tmp_img)
                    tmp_img = np.where(tmp_img < 0, 0, tmp_img).astype(np.uint8)
                cnt += 1

                tmp_image.append(tmp_img)
                tmp_label.append(label)

        train_image.extend(tmp_image)
        train_label.extend(tmp_label)

        print(d + 'read complete ,' + str(len(train_label)) + ' pictures exit')


    plt.figure(figsize=(10, 10))
    data_len =  len(train_image)

    if mode == 'check':
        converted_image = np.array(train_image)[:, :, :, ::-1]
        for i in range(9):
            index = random.randint(0, data_len)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(converted_image[index])
            plt.title(label_list[train_label[index].argmax()])
            plt.axis("off")
        plt.show()
        exit()

    X_train = np.array(train_image)/255.0
    y_train = np.array(train_label)

    if split_num != 0 and len(X_train) <= split_num:
        X_train, y_train, _, _ = train_test_split(X_train, y_train, train_size = split_num, shuffle = shuffle_flag)

    return X_train , y_train


def gen_data(mode = '', img_up_num = 1, img_noise = 0 , img_size = 0, clip_size = 0 ,img_rotate = 0, img_flip = 0, shuffle_flag = True):
    clear_data()
    if not os.path.exists(download_path):
        extract_dataset(url, download_path)

    if not os.path.exists(data_path + data_file_name + data_list[-1] + data_list[-1] + '_data.npz'):
        split_num_list = [0, 0, 0]
        i = 0
        for data_type in data_list:
            data_directory = data_path + data_file_name + data_type
            if data_type == '/test':
                image, label = make_data(mode = mode,
                                         folder_name = data_directory,
                                         split_num = split_num_list[i],
                                         img_up_num = 1,
                                         img_noise = img_noise,
                                         img_size = img_size,
                                         clip_size = clip_size,
                                         img_rotate = img_rotate,
                                         img_flip = img_flip,
                                         shuffle_flag = False)
            elif data_type == '/valid':
                image, label = make_data(mode = mode,
                                         folder_name = data_directory,
                                         split_num = split_num_list[i],
                                         img_up_num = 1,
                                         img_noise = img_noise,
                                         img_size = img_size,
                                         clip_size = clip_size,
                                         img_rotate = img_rotate,
                                         img_flip = img_flip,
                                         shuffle_flag = shuffle_flag)
            else:
                image, label = make_data(mode = mode,
                                         folder_name = data_directory,
                                         split_num = split_num_list[i],
                                         img_up_num = img_up_num,
                                         img_noise = img_noise,
                                         img_size = img_size,
                                         clip_size = clip_size,
                                         img_rotate = img_rotate,
                                         img_flip = img_flip,
                                         shuffle_flag = shuffle_flag)

            np.savez(data_directory + data_type + '_data', image, label)
            i += 1

def data_load(data_type):
    load_data = np.load(data_path + data_file_name + data_type + data_type + '_data.npz', mmap_mode = 'r+', allow_pickle=True)
    X = load_data['arr_0']
    y = load_data['arr_1']

    return X, y
