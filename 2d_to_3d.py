#dlib_xy + gan_depth
import numpy as np
fr = open('./face_alignmented/keypoints_text/source.png.txt','r')
xy_list = fr.readlines()
ww = open('./face_alignmented/keypoints_text/3d_keypoints_txt', 'w')
count = -1
ct = -1
x = np.load('./face_alignmented/keypoints_text/depth.npy')
for xy in xy_list:
    
    count+=1
    if count == 60:
        continue
    if count == 64:
        continue
    ct+=1
    print(xy.split(' ')[0]+ ' ' +xy.split(' ')[1][:-1]+' '+str(x[ct])+'\n')
    print("count:",count)
    ww.write(xy.split(' ')[0]+ ' ' +xy.split(' ')[1][:-1]+' '+str(x[ct])+'\n')
    
ww.close()
