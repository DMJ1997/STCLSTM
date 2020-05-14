import os
import cv2
DATADIR="./nyu2_train"
DATADIR_SAVE="./data2"

path=os.path.join(DATADIR) 
save_path=os.path.join(DATADIR_SAVE) 
img_list=os.listdir(path)
for x in img_list:
    path2 = os.path.join(path,x)
    imglist2 = os.listdir(path2)
    for y in imglist2:
        img_array=cv2.imread(os.path.join(path2,y),cv2.IMREAD_COLOR)
        new_array=cv2.resize(img_array,(320,240))
        save_path=os.path.join(save_path,x)
        save_path=os.path.join(save_path,y)
        cv2.imwrite(save_path,new_array)
