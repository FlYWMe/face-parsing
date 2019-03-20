import cv2
import torch.utils.data

class MyDataset(torch.utils.data.Dataset):
	def __init__(self,imlist,labelist,transform=None):
		self.imlist=imlist
		self.labelist=labelist
		self,transform=transform

	def __len__(self):
		return len(self.imlist)

	def __getitem__(self,idx):
		imagename=self.imlist[idx]
		labelname=self.labelist[idx]
		imag=cv2.imread(imagename)
		label=cv2.imread(labelname)
		if self.transform:
			[imag,label]= self.transform(imag,label)
		return (imag, label)