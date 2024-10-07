import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import random as rn
import matplotlib.pyplot as plt

dataset_path = "./data/datasets.npz"
seed = 1234
train_size  = 0.8
class_num = 10
metric = 'val_loss'
checkpoint_mode = 'min'
model_path = './model/task01_'


tf.random.set_seed(seed)
np.random.seed(seed)
rn.seed(seed)

def data_loader():

    if not os.path.exists(dataset_path):
        data, info = tfds.load("imagenette/160px-v2", with_info=True, as_supervised=True)
        train_data, test_data = data['train'], data['validation']

        del data

        train_dataset = train_data.map(
            lambda image, label: (tf.image.resize(image, (81, 81)), label))

        test_dataset = test_data.map(
            lambda image, label: (tf.image.resize(image, (81, 81)), label)
        )

        del train_data
        del test_data

        num_classes = info.features['label'].num_classes

        X_train = np.array(list(map(lambda x: x[0], train_dataset)))
        y_train = np.array(list(map(lambda x: x[1], train_dataset)))


        X_test = np.array(list(map(lambda x: x[0], test_dataset)))
        y_test = np.array(list(map(lambda x: x[1], test_dataset)))

        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        del train_dataset
        del test_dataset


        np.savez(dataset_path, X_train, y_train, X_test, y_test)

    else:
        load_data = np.load(dataset_path)

        X_train = load_data['arr_0']
        y_train = load_data['arr_1']
        X_test = load_data['arr_2']
        y_test = load_data['arr_3']

        del load_data

    return X_train, y_train, X_test, y_test

def test(current_model_path, X_test, y_test):
    load_model = tf.keras.models.load_model(current_model_path)

    print("Evaluate on test data")
    results = load_model.evaluate(X_test, y_test)
    print("test loss", results[0])
    print("test accuracy", results[1])
    if results[1] >= 0.5:
        print("Congratulations! You've completed this task!")
    else:
        print("You missed our goal. Let's do our best!")

    return 0       


def show_datasets():

    data, info = tfds.load("imagenette/160px-v2", with_info=True, as_supervised=True)
    train_data = data['train']

    train_dataset = train_data.map(
    lambda image, label: (tf.image.resize(image, (160, 160)), label))
    num_classes = info.features['label'].num_classes

    del train_data

    get_label_name = info.features['label'].int2str
    text_labels = [get_label_name(i) for i in range(num_classes)]

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      height_shift_range=0.2)

    X_train = list(map(lambda x: x[0], train_dataset))
    y_train = list(map(lambda x: x[1], train_dataset))

    train_ds = tf.keras.preprocessing.image.NumpyArrayIterator(
    x=np.array(X_train), y=np.array(y_train), image_data_generator=train_datagen, batch_size=16
    )

    image, label = next(iter(train_ds))
    plt.imshow(image[0])
    # plt.title(get_label_name(np.argmax(label)))

    for i in range(0, 9):
        image, label = next(iter(train_ds))
        plt.subplot(330 + 1 + i)
        plt.imshow(image[0])
        #plt.title(get_label_name(np.argmax(label)))
    # show the plot
    plt.show()

    return 0


def draw_learning_curve(history, time_stamp, epoch):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend(frameon=False)
    plt.xlabel("epochs")
    plt.ylabel("crossentropy")

    plt.savefig("loss_" + str(epoch) + "_" + time_stamp + ".png")
    plt.show()

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.legend(frameon=False)
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")

    plt.savefig("loss_" + str(epoch) + "_" + time_stamp + ".png")
    plt.show()

