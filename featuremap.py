from keras.models import load_model
import numpy as np
from keras import backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import cv2
import matplotlib.pyplot as plt


# モデルの読込み
model = load_model('results_150pt/finetuning.h5')
# model.summary()


# 対象イメージの読込み
jpg_name = 'umi/0_2_52'
img_path = ('dataset/validation/face_150/' + jpg_name + '.jpg')
img = img_to_array(load_img(img_path, target_size=(150,150)))
H,W = img.shape[:2]
img_nad = img_to_array(img)/255
img_nad = img_nad[None, ...]


# 特長マップを抜き出すレイヤー指定
get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[17].output])
print(model.layers[17])
layer_output = get_layer_output([img_nad, 0])[0]

# 特長マップ合成
G, R, ch = layer_output.shape[1:]
res = np.zeros((G,R))

for i in range(ch):
    img_res = layer_output[0,:,:,i]
    res = res + img_res

res = res/ch

# 特長マップ平均の平坦化
res_flatte = np.ma.masked_equal(res,0)
res_flatte = (res_flatte - res_flatte.min())*255/(res_flatte.max()-res_flatte.min())
res_flatte = np.ma.filled(res_flatte,0)

# 色付け
acm_img = cv2.applyColorMap(np.uint8(res_flatte), cv2.COLORMAP_JET)
acm_img = cv2.cvtColor(acm_img, cv2.COLOR_BGR2RGB)
acm_img = cv2.resize(acm_img,(H,W))

# 元絵と合成
mixed_img = (np.float32(acm_img)*0.6 + img *0.4)

# 表示
out_img = np.concatenate((img, acm_img, mixed_img), axis=1)
plt.imshow(array_to_img(out_img))
plt.show()
