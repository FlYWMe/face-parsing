#encoding=utf-8

import numpy as np 
import torch
# import r18unet
from torch.autograd import Variable
import glob
import cv2
import sys
# import unet
import os
from PIL import Image
import mobileunet
def segvideo():
    pallete=[0, 0, 0,
        0, 255 ,255,  #face
        57, 50, 200,  #left eyebrow
        38, 54, 139,  #r eyebrow
        0, 55, 100,   #l eye
        0, 138, 0,    #r eye
        205, 50, 154, #nose
        139, 0, 0,    #upper lip
        0, 165, 255,  #in mouth
        50, 0, 200,   #lower lip
        255, 118, 72]; #hair
        
    # model= unet.UNet(11)
    # model = r18unet.ResNetUNet(11)
    model = mobileunet.MobileUNet(11)
    save_path = 'model_80.pth'
    state_dict = torch.load(save_path)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k[7:]
        new_state_dict[namekey] = v

    model.load_state_dict(new_state_dict )
    model = model.cuda()
    model.eval()

    # cap = cv2.VideoCapture('videoface.mp4')
    cap = cv2.VideoCapture('20.mp4')
    # cap.open('videoface.mp4')
    print(cap.isOpened())
     
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    imgw =size[0]
    imgh =size[1]
    print(size)
    print(size[0],size[1])
    #fps是帧率，意思是每一秒刷新图片的数量，frames是一整段视频中总的图片数量。
    print("fps=",fps,"frames=",frames)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videoWriter = cv2.VideoWriter('seg.mp4', fourcc, fps,(size[0]*2,size[1])) # 5th is RGB/GRAY
    i=0
    while(True):
        i+=1
        # print(i)
        ret, frame = cap.read()
        # print(frame.dtype)  uint8
        # print(frame.shape)  360,636,3

        if ret:
            img= cv2.resize(frame, (224,224)).astype(np.float32)
            img /= 255
            img = img.transpose((2,0,1))
            img_tensor = torch.from_numpy(img)
            img_tensor = torch.unsqueeze(img_tensor, 0)
            img_variable = Variable(img_tensor).cuda()
            img_out = model(img_variable)
            # print(img_out.shape)  1 11 224 224
     
            img_out_norm = torch.squeeze(img_out, 0)
            prob, classmap = torch.max(img_out_norm, 0)
           
            classmap_np = classmap.data.cpu().numpy()
            # print(type(classmap_np))
            im_pil = Image.fromarray(np.uint8(classmap_np))
            # print(im_pil.mode)  # mode :L gray 
            # print(im_pil.size)  224
            im_pil.putpalette(pallete)  # mode->p
            # print(im_pil.size)  224

            im_pil = im_pil.resize((size[0],size[1]) ,Image.ANTIALIAS)
            im_pil = im_pil.convert('RGB')  #p->rgb (follow  pallete bgr)

            # im_pil.save("res/"+str(i)+".png")
            im_pil_np = np.asarray(im_pil)  #, dtype=np.float32)
            # print(im_pil_np.dtype)
            # print(im_pil_np.size)
            seg = 0.35*frame + 0.65*im_pil_np
            # print(frame.dtype)
            # print(seg.dtype)
            merge = np.hstack((frame,seg.astype(np.uint8)))
            videoWriter.write(merge)
            # break
        else:
            break
    cap.release()
    videoWriter.release()


def mergevideo():
    left = cv2.VideoCapture('20.mp4')
    right = cv2.VideoCapture('out1.mp4')
    fps = left.get(cv2.CAP_PROP_FPS)
    width = (int(left.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = (int(left.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    videoWriter = cv2.VideoWriter('merge.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width*2, height))
    
    i = 0
    while True:
        i+=1
        print(i)
        lret, lframe = left.read()
        rret, rframe = right.read()
        if lret and rret:
            framemerge = np.hstack((lframe, rframe))
            videoWriter.write(framemerge)
        else:
            break
    videoWriter.release()
    left.release()
    right.release()

segvideo()
# mergevideo()