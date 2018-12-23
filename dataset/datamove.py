import os
import shutil
import random

ori_dir = 'train/face_150/'
ter_dir = 'validation/face_150/'

face_dirs = os.listdir(ori_dir)
print(face_dirs)
for face_dir in face_dirs:
    if not os.path.isdir(ori_dir+face_dir):
        continue
    files = os.listdir(ori_dir+face_dir)
    num = round(len(files)*15/85)
    print([face_dir, len(files), num])  # 名前、移動前、移動数
    if not os.path.exists(ter_dir+'/'+face_dir):
        os.makedirs(ter_dir+'/'+face_dir)
    chosen_files = random.sample(files, num)
    for file in chosen_files:
        shutil.move(ori_dir+face_dir+'/'+file, ter_dir+face_dir+'/'+file)

