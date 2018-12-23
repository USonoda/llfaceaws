import cv2
import glob
import os

lllist = ['高坂穂乃果', '南ことり', '園田海未', '小泉花陽', '星空凛', '西木野真姫', '矢澤にこ', '絢瀬絵里', '東條希']
classes = ['honoka', 'kotori', 'umi', 'hanayo', 'rin', 'maki', 'nico', 'eli', 'nozomi']
others = ["きんモザ", "てーきゅう", "ラブライブサンシャイン", "ごちうさ", "りゅうおうのおしごと!"]
others2 = ["アイマス", "sideM", "アイカツ", "けいおん！"]
search_list = classes + others + others2


img_width, img_height = 150, 150

result = {}
for member in search_list:
    # 元画像を取り出して顔部分を正方形で囲み、指定サイズにリサイズ、別ファイルに保存
    in_dir = './' + member + "/*"
    out_dir_name = 'face/' + member
    if not os.path.exists(out_dir_name):
        os.makedirs(out_dir_name)
    out_dir = "./" + out_dir_name
    print('============================================================================================')
    print(in_dir)
    in_jpg = glob.glob(in_dir)
    in_fileName = os.listdir("./"+member)
    # print(os.listdir("."))
    # print(in_jpg)
    # print(in_fileName)
    print(len(in_jpg))
    face_num = 0
    for num in range(len(in_jpg)):
        image = cv2.imread(str(in_jpg[num]))
        if image is None:
            print("Not open:")
            continue

        image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier("lbpcascade_animeface/lbpcascade_animeface.xml")
        # 顔認識の実行
        face_list = cascade.detectMultiScale(image_gs,
                                             scaleFactor=1.1, minNeighbors=2,
                                             minSize=(int(img_width/2), int(img_height/2)))
        # 顔が１つ以上検出された時
        if len(face_list) > 0:
            print('face number is '+str(len(face_list)))

            count = 0
            for rect in face_list:
                # print('count is '+str(count))
                # print('rect: '+str(rect))
                x, y, width, height = rect
                face_image = image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
                if rect[2] < img_width:
                    print('too small')
                    continue
                face_image = cv2.resize(face_image, (img_width, img_height))

                print(face_image.shape)
                # 保存
                fileName = os.path.join(out_dir, str(count)+'_'+str(in_fileName[num]))
                try:
                    cv2.imwrite(str(fileName), face_image)
                    count += 1
                    face_num += 1
                except cv2.error:
                    print('extension err')
        # 顔が検出されなかった時
        else:
            print("no face")
            continue

    print(member+": "+str(face_num)+"枚")
    result[member] = face_num

print(result)

# {'honoka': 295, 'kotori': 294, 'umi': 279, 'hanayo': 312, 'rin': 289,
# 'maki': 253, 'nico': 281, 'eli': 287, 'nozomi': 309,
#  'きんモザ': 96, 'てーきゅう': 70, 'ラブライブサちうさ': 100, 'りゅうおうのおしごと!': 77,
# 'アイマス': 67, 'sideM': 82, 'アイカツ': 55, 'けいおん！': 89}

