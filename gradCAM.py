import numpy as np
import cv2
from keras import backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import load_model
import matplotlib.pyplot as plt
import os
from random import shuffle

K.set_learning_phase(1)  # set learning phase


def evaluation(img_path):
    filename = img_path
    img = load_img(filename, target_size=(150,150))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255
    pred = model.predict(x)[0]
    top_indices = pred.argsort()[-3:][::-1]

    result = [(classes[i], pred[i]) for i in top_indices]
    return result


def Grad_Cam(x, layer_name):
    """
    Args:
       x: 画像(array)
       layer_name: 畳み込み層の名前

    Returns:
       jetcam: 影響の大きい箇所を色付けした画像(array)

    """

    # 前処理
    X = np.expand_dims(x, axis=0)

    X = X.astype('float32')
    preprocessed_input = X / 255.0

    # 予測クラスの算出
    predictions = model.predict(preprocessed_input)
    class_idx = np.argmax(predictions[0])
    class_output = model.output[:, class_idx]

    #  勾配を取得
    conv_output = model.get_layer(layer_name).output  # layer_nameのレイヤーのアウトプット
    grads = K.gradients(class_output, conv_output)[0]  # gradients(loss, variables) で、variablesのlossに関しての勾配を返す
    gradient_function = K.function([model.input], [conv_output, grads])  # model.inputを入力すると、conv_outputとgradsを出力する関数

    output, grads_val = gradient_function([preprocessed_input])
    output, grads_val = output[0], grads_val[0]

    # 重みを平均化して、レイヤーのアウトプットに乗じる
    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # 画像化してヒートマップにして合成
    cam = cv2.resize(cam, (150, 150), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # モノクロ画像に疑似的に色をつける
    jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)  # 色をRGBに変換
    jetcam = (np.float32(jetcam) + x / 2)   # もとの画像に合成

    return jetcam


classes = ['honoka', 'kotori', 'umi', 'hanayo', 'rin', 'maki', 'nico', 'eli', 'nozomi','others']
model = load_model('results_150pt/finetuning.h5')
img_dir = 'dataset/validation/face_150/nozomi/'
# img_dir = 'dataset/test3/'

n = 3  # n**2個の画像が表示される


i,c = 0,1
fig = plt.figure()
l = os.listdir(img_dir)
shuffle(l)
while True:
    if c > min(len(l), n**2) or i == len(l):
        break
    print(c)
    img_path = (img_dir + l[i])
    if l[i] == '.DS_Store':
        i += 1
        continue

    x = img_to_array(load_img(img_path, target_size=(150, 150)))
    array_to_img(x)
    image = Grad_Cam(x, 'block5_conv3')

    ax = fig.add_subplot(n, n, c)
    ax.imshow(array_to_img(image))
    plt.axis("off")
    name, num = evaluation(img_path)[0]
    ax.add_patch(plt.Rectangle(xy=[110,125], width=50, height=40, color='white', alpha=0.8))
    plt.text(115, 135, name, fontsize=5)
    plt.text(115, 150, round(num,3), fontsize=5)
    i += 1; c += 1
plt.show()
