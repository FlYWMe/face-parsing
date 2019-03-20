import torch
import torch.nn.functional as F
from torch import nn
import utils 


class EncoderBlock(nn.Module):
	def __init__(self,in_channel,out_channel,dropout=False):
		super(EncoderBlock, self).__init__()
		# print(in_channel, out_channel)

		layers=[
			nn.Conv2d(in_channel,out_channel,kernel_size=3),
			nn.BatchNorm2d(out_channel),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channel,out_channel,kernel_size=3),
			nn.BatchNorm2d(out_channel),
			nn.ReLU(inplace=True),
		]
		if(dropout):
			layers.append(nn.Dropout())
		layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
		self.encode=nn.Sequential(*layers)

	def forward(self,x):
		return self.encode(x)

class DecoderBlock(nn.Module):
	def __init__(self, in_channel,middle_channels, out_channel):
		super(DecoderBlock, self).__init__()
		layers=[
			nn.Conv2d(in_channel,middle_channels,kernel_size=3),
			nn.BatchNorm2d(middle_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(middle_channels,middle_channels,kernel_size=3),
			nn.BatchNorm2d(middle_channels),
			nn.ReLU(inplace=True),
			nn.ConvTranspose2d(middle_channels,out_channel,kernel_size=2,stride=2),
		]
		self.decode=nn.Sequential(*layers)

	def forward(self,x):
		return self.decode(x)

class UNet(nn.Module):
	def __init__(self, num_classes):
		super(UNet,self).__init__()
		self.en1=EncoderBlock(3,64)
		self.en2=EncoderBlock(64,128)
		self.en3=EncoderBlock(128,256)
		self.en4=EncoderBlock(256,512,dropout=True)

		self.center=DecoderBlock(512,1024,512)
		self.de4=DecoderBlock(1024,512,256)
		self.de3=DecoderBlock(512,256,128)
		self.de2=DecoderBlock(256,128,64)
		self.de1=nn.Sequential(
			nn.Conv2d(128,64,kernel_size=3),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(64,64,kernel_size=3),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
		)
		self.final=nn.Conv2d(64,num_classes,kernel_size=1)
		# utils.initialize_weights(self)

	def forward(self,x):
		# print("UNet running...")
		# print("x: ", x.size())
		en1=self.en1(x)
		# print(en1.shape)
		en2=self.en2(en1)
		# print(en2.shape)
		en3=self.en3(en2)
		# print(en3.shape)
		en4=self.en4(en3)
		# print(en4.shape)
		center=self.center(en4)
		# print(center.shape)

		de4=self.de4(torch.cat([center, F.upsample(en4,center.size()[2:], mode='bilinear')] ,1))
		# print(de4.shape,F.upsample(en3,de4.size()[2:], mode='bilinear').shape)
		de3=self.de3(torch.cat([de4, F.upsample(en3,de4.size()[2:], mode='bilinear')] ,1))
		# print(de3.shape)
		de2=self.de2(torch.cat([de3, F.upsample(en2,de3.size()[2:], mode='bilinear')] ,1))
		# print(de2.shape)
		de1=self.de1(torch.cat([de2, F.upsample(en1,de2.size()[2:], mode='bilinear')] ,1))
		# print(de1.shape)

		final=self.final(de1)
		# print(final.shape,'final')
		# print(x.shape,'x')
		ret= F.upsample(final, x.size()[2:], mode='bilinear')
		# print(ret.shape)
		return ret
