import os, sys
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
from keras import optimizers
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

classes = ['honoka', 'kotori', 'umi', 'hanayo', 'rin', 'maki', 'nico', 'eli', 'nozomi', 'others']
nb_classes = len(classes)
img_width, img_height = 150, 150

result_dir = 'results_all/'

# このディレクトリにテストしたい画像を格納しておく
test_data_dir = 'dataset/test'


def model_load():
    # VGG16, FC層は不要なので include_top=False
    input_tensor = Input(shape=(img_width, img_height, 3))
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    # FC層の作成
    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(nb_classes, activation='softmax'))

    # VGG16とFC層を結合してモデルを作成
    model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

    # 学習済みの重みをロード
    model.load_weights(os.path.join(result_dir, 'finetuning.h5'))

    # 多クラス分類を指定
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

    return model


def evaluation(img_path):
    # モデルのロード
    model = load_model(result_dir)

    filename = img_path
    img = image.load_img(filename, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # 学習時に正規化してるので、ここでも正規化
    x /= 255
    pred = model.predict(x)[0]

    top = 3  # 上位3つを表示
    top_indices = pred.argsort()[-top:][::-1]
    result = [(classes[i], pred[i]) for i in top_indices]

    return result


if __name__ == '__main__':
    model = load_model(result_dir+'finetuning.h5')
    test_imagelist = os.listdir(test_data_dir)

    for test_image in test_imagelist:
        if test_image == '.DS_Store':
            continue
        filename = os.path.join(test_data_dir, test_image)
        img = image.load_img(filename, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255
        pred = model.predict(x)[0]

        top = 3 #上位3つを表示
        top_indices = pred.argsort()[-top:][::-1]
        result = [(classes[i], pred[i]) for i in top_indices]
        print('=======================================')
        print('file name is', test_image)
        print(result)
        print('=======================================')
