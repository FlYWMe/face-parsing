import numpy as np
import torch
import random
import cv2

class RandomHorizontalFlip(object):
	def __call__(self,image,label):
		# w,h = image.shape[:2]
		# print("w: ",w)
		# print("h: ",h)
		# if random.random() <0.5:
		# 	x1=random.randint(0,1)
		# 	if x1==0:
		# 		image=cv2.flip(image,0)
		# 		label=cv2.flip(label,0)
		# 	else:
		image=cv2.flip(image,1)
		label=cv2.flip(label,1)

		return [image,label]

class ToTensor(object):
	def __init__(self,scale=1):
		self.scale=scale
	def __call__(self,image,label):
		if self.scale!=1:
			w,h = label.shape[:2]
			label = cv2.resize(label,(int(w/self.scale),int(h/self.scale)),interpolation=cv2.INTER_NEAREST)
		image= image.transpose((2,0,1))
		image= image.astype(np.float32)

		image_tensor= torch.from_numpy(image).div(255)
		label_tensor= torch.LongTensor(np.array(label,dtype=np.int))
		return [image_tensor,label_tensor]

class RandomCropResize(object):
	def __init__(self, inwidth,inheight):
		self.cw= inwidth
		self.ch= inheight

	def __call__(self, img, label):

		img_crop = cv2.resize(img, (self.cw,self.ch))
		label_crop = cv2.resize(label, (self.cw, self.ch), interpolation=cv2.INTER_NEAREST)
		return img_crop, label_crop
		# if random.random() <0.5:
		# 	w,h = img.shape[:2]
		# 	x1 = random.randint(0, self,ch)
		# 	y1 = random.randint(0, self,cw)
		# 	img_crop = img[y1:h - y1, x1:w-x1]
		# 	label_crop = label[y1:h - y1, x1:w-x1]
		# 	img_crop = cv2.resize(img_crop, (mysize,mysize))
		# 	label_crop = cv2.resize(label_crop, (mysize,mysize), interpolation=cv2.INTER_NEAREST)
		# 	return img_crop, label_crop
		# else:
		# 	return [img,label]

class Compose(object):
	def __init__(self, transforms):
		self.transforms = transforms

	def __call__(self, *args):
		for t in self.transforms:
			args = t(*args)
		return args