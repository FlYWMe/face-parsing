import numpy as np 
import torch
import r18unet
from torch.autograd import Variable
import glob
import cv2
import sys
import unet
import os
from PIL import Image
import mobileunet

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

pallete=[0, 0, 0,
    255, 255 ,0,
    139, 76, 57,
    139, 54, 38,
    0, 205, 0,
    0, 138, 0,
    154, 50, 205,
    0, 0, 139,
    255, 165, 0,
    72, 118, 255,
    238, 154, 0];
# model= unet.UNet(11)
# model = r18unet.ResNetUNet(11)
model = mobileunet.MobileUNet(11)
save_path = '../result/224-adam-mobile/model_70.pth'
state_dict = torch.load(save_path)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
	namekey = k[7:]
	new_state_dict[namekey] = v

model.load_state_dict(new_state_dict )
model = model.cuda()
model.eval()

image_list = glob.glob('../data/test/*.jpg')
# print(image_list)
outdir = '../test/'
for imgname in image_list:
	img= cv2.imread(imgname)
	imgw = img.shape[0]
	imgh = img.shape[1]
	# print(type(img))
	img= cv2.resize(img, (224,224)).astype(np.float32)
	# print(type(img))
	img /= 255
	img = img.transpose((2,0,1))
	img_tensor = torch.from_numpy(img)
	img_tensor = torch.unsqueeze(img_tensor, 0)
	img_variable = Variable(img_tensor).cuda()
	# print(img_variable.shape)
	img_out = model(img_variable)
	print(img_out.shape)

	img_out_norm = torch.squeeze(img_out, 0)
	# print(img_out_norm)
	prob, classmap = torch.max(img_out_norm, 0)
	classmap_np = classmap.data.cpu().numpy()
	print(classmap_np.shape)
	im_pil = Image.fromarray(np.uint8(classmap_np))
	im_pil.putpalette(pallete)
	print(imgname)
	name = imgname.split('/')[-1][:-3]+'png'
	 
	im_pil = im_pil.resize((imgh,imgw) ,Image.ANTIALIAS)
	im_pil.save(outdir +name)