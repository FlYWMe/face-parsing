#coding=utf-8
import torch
import numpy as np

class iouEval:
	def __init__(self, classes):
		self.classes=classes
		self.reset()

	def reset(self):
		self.overall_acc=0
		self.per_class_acc=np.zeros(self.classes, dtype=np.float32)
		self.per_class_iu=np.zeros(self.classes, dtype=np.float32)
		self.mIOU=0
		self.batchCount=1

	def compute_hist(self,pred,gth):
		k=(gth>=0) & (gth<self.classes)
		# k=[True, True, True,...] bincount return:[类混淆矩阵]classes*classes
		return np.bincount(self.classes*gth[k].astype(int) + pred[k], minlength=self.classes**2).reshape(self.classes,self.classes)

	def addBatch(self, pred,gth):
		pred=pred.cpu().numpy().flatten()
		gth=gth.cpu().numpy().flatten()
		epsilon = 0.00000001
		hist=self.compute_hist(pred, gth)
		overall_acc=np.diag(hist).sum() / (hist.sum()+epsilon)
		per_class_acc=np.diag(hist) / (hist.sum(1)+epsilon)
		per_class_iu=np.diag(hist) / (hist.sum(1) + hist.sum(0) -np.diag(hist)+epsilon)
		mIou=np.nanmean(per_class_iu)
		self.overall_acc+=overall_acc
		self.per_class_acc+=per_class_acc
		self.per_class_iu+=per_class_iu
		self.mIOU+=mIou
		self.batchCount+=1

	def getMetric(self):
		overall_acc=self.overall_acc/self.batchCount
		per_class_acc=self.per_class_acc/self.batchCount
		per_class_iu=self.per_class_iu/self.batchCount
		mIOU=self.mIOU/self.batchCount

		return  overall_acc,per_class_acc,per_class_iu,mIOU