import loadData as ld
import os
import torch
import pickle
import unet 
import Model 
from torch.autograd import Variable
import VisGraph as viz
from Criteria import CrossEntropyLoss2d
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler
import Transforms as myTransforms
# import DataSet as myDataLoader
import time
from argparse import ArgumentParser
from IOUEval import iouEval
import numpy as np
import r18unet
import mobileunet
from torch.nn.functional import interpolate

def val(args,val_loader,model,criteria):
	model.eval()
	iouEvalVal=iouEval(args.classes)
	epoch_loss=[]
	total_batches=len(val_loader)
	for i, (input,target) in enumerate(val_loader):
		start_time=time.time()
		if args.onGPU==True:
			input = input.cuda()
			target=target.cuda()
		input_var=torch.autograd.Variable(input,volatile=True)
		target_var=torch.autograd.Variable(target,volatile=True)
		output=model(input_var)

		loss=criteria(output, target_var)
		
		epoch_loss.append(loss.item())
		time_taken=time.time()-start_time

		iouEvalVal.addBatch(output.max(1)[1].data, target_var.data)
		# print('[%d/%d] loss: %.3f time:%.2f' % (i, total_batches, loss.item(), time_taken))

	average_epoch_loss_val=sum(epoch_loss)/ len(epoch_loss)
	overall_acc,per_class_acc,per_class_iu,mIOU=iouEvalVal.getMetric()

	return average_epoch_loss_val,overall_acc,per_class_acc,per_class_iu,mIOU



def train(args,train_loader,model,criteria,optimizer,epoch):
	model.train()
	iouEvalTrain=iouEval(args.classes)
	epoch_loss=[]
	total_batches=len(train_loader)
	# print(total_batches)

	for i, (input,target) in enumerate(train_loader):
		# print("input: ", input.size()[:], target.size()[:], input.size(0))
		start_time=time.time()
		if args.onGPU==True:
			input = input.cuda()
			target=target.cuda()
		input_var=torch.autograd.Variable(input)
		target_var=torch.autograd.Variable(target)
		output=model(input_var)
		# print('==================')
		# print(output.shape)
		# print(target_var.shape)
		# print('==================')
		optimizer.zero_grad()
		loss=criteria(output, target_var)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		epoch_loss.append(loss.item())  #loss.data[0]
		time_taken=time.time()-start_time

		iouEvalTrain.addBatch(output.max(1)[1].data, target_var.data)
		# print('[%d/%d] loss: %.3f time:%.2f' % (i, total_batches, loss.item(), time_taken))

	average_epoch_loss_train=sum(epoch_loss)/ len(epoch_loss)
	overall_acc,per_class_acc,per_class_iu,mIOU=iouEvalTrain.getMetric()

	return average_epoch_loss_train,overall_acc,per_class_acc,per_class_iu,mIOU


def trainValSegmentation(args):
	if not os.path.isfile(args.cached_data_file):
		dataLoader=ld.LoadData(args.data_dir,args.classes,args.attrClasses,args.cached_data_file)
		if dataLoader is None:
			print("Error while cacheing the data.")
			exit(-1)
		data = dataLoader.processData()
	else:
		print("load cacheing data.")
		data= pickle.load(open(args.cached_data_file,'rb'))
	# only unet for segmentation now.
	# model= unet.UNet(args.classes)
	# model = r18unet.ResNetUNet(args.classes)
	model = mobileunet.MobileUNet(args.classes)
	print("UNet done...")
	# if args.onGPU == True:
	model=model.cuda()
	# devices_ids=[2,3], device_ids=range(2)
	# device = torch.device('cuda:' + str(devices_ids[0]))
	# model = model.to(device)
	if args.visNet == True:
		x = Variable(torch.randn(1,3,args.inwidth, args.inheight))
		if args.onGPU == True:
			x=x.cuda()
		print("before forward...")
		y = model.forward(x)
		print("after forward...")
		g = viz.make_dot(y)
		# g1 = viz.make_dot(y1)
		g.render(args.save_dir + '/model', view=False)
	model = torch.nn.DataParallel(model)
	n_param = sum([np.prod(param.size()) for param in model.parameters()])
	print('network parameters: ' + str(n_param))

	#define optimization criteria
	weight = torch.from_numpy(data['classWeights'])
	print(weight)
	if args.onGPU == True:
		weight = weight.cuda()
	criteria = CrossEntropyLoss2d(weight)
	# if args.onGPU == True:
	# 	criteria = criteria.cuda()

	trainDatasetNoZoom = myTransforms.Compose([
			myTransforms.RandomCropResize(args.inwidth,args.inheight),
			# myTransforms.RandomHorizontalFlip(),
			myTransforms.ToTensor(args.scaleIn)
		])
	trainDatasetWithZoom = myTransforms.Compose([
			# myTransforms.Zoom(512,512),
			myTransforms.RandomCropResize(args.inwidth,args.inheight),
			myTransforms.RandomHorizontalFlip(),
			myTransforms.ToTensor(args.scaleIn)
		])
	valDataset = myTransforms.Compose([
			myTransforms.RandomCropResize(args.inwidth,args.inheight),
			myTransforms.ToTensor(args.scaleIn)
		])
	trainLoaderNoZoom = torch.utils.data.DataLoader(
		ld.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDatasetNoZoom),
		batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
	trainLoaderWithZoom = torch.utils.data.DataLoader(
		ld.MyDataset(data['trainIm'], data['trainAnnot'], transform=trainDatasetWithZoom),
		batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
	valLoader = torch.utils.data.DataLoader(
		ld.MyDataset(data['valIm'], data['valAnnot'], transform=valDataset),
		batch_size=args.batch_size_val, shuffle=True, num_workers=args.num_workers, pin_memory=True)

	#define the optimizer
	optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.999), eps=1e-08, weight_decay=2e-4)
	# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.99, weight_decay=5e-4)
	# optimizer = torch.optim.SGD([
 #        {'params': [param for name, param in model.named_parameters() if name[-4:] == 'bias'],
 #         'lr': 2 * args.lr},
 #        {'params': [param for name, param in model.named_parameters() if name[-4:] != 'bias'],
 #         'lr': args.lr, 'weight_decay': 5e-4}
 #    ], momentum=0.99)

	if args.onGPU == True:
		cudnn.benchmark = True
	start_epoch=0
	if args.resume:
		if os.path.isfile(args.resumeLoc):
			print("=> loading checkpoint '{}'".format(args.resumeLoc))
			checkpoint = torch.load(args.resumeLoc)
			start_epoch=checkpoint['epoch']
			model.load_state_dict(checkpoint['state_dict'])
			print("=> loaded checkpoint '{}' (epoch{})".format(args.resume, checkpoint['epoch']))
		else:
			print("=> no checkpoint found at '{}'".format(args.resumeLoc))
	
	logfileLoc= args.save_dir+ os.sep+args.logFile
	print(logfileLoc)
	if os.path.isfile(logfileLoc):
		logger=open(logfileLoc,'a')
		logger.write("parameters: %s" % (str(n_param)))
		logger.write("\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t" % ('Epoch', 'Loss(Tr)', 'Loss(val)','Overall acc(Tr)','Overall acc(val)', 'mIOU (tr)', 'mIOU (val'))
		logger.flush()
	else:
		logger = open(logfileLoc, 'w')
		logger.write("Parameters: %s" % (str(n_param)))
		logger.write("\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t" % ('Epoch', 'Loss(Tr)', 'Loss(val)','Overall acc(Tr)','Overall acc(val)', 'mIOU (tr)', 'mIOU (val'))
		logger.flush()

	#lr scheduler
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90],gamma=0.1)
	best_model_acc=0
	for epoch in range(start_epoch, args.max_epochs):
		scheduler.step(epoch)
		lr=0
		for param_group in optimizer.param_groups:
			lr= param_group['lr']
		# train(args,trainLoaderWithZoom,model,criteria,optimizer,epoch)
		lossTr,overall_acc_tr,per_class_acc_tr,per_class_iu_tr,mIOU_tr=train(args,trainLoaderNoZoom,model,criteria,optimizer,epoch)
		# print(per_class_acc_tr,per_class_iu_tr)
		lossVal, overall_acc_val, per_class_acc_val, per_class_iu_val, mIOU_val = val(args, valLoader, model, criteria)

		#save_checkpoint
		torch.save({
			'epoch': epoch+1,
			'arch': str(model),
			'state_dict': model.state_dict(),
			'optimizer': optimizer.state_dict(),
			'lossTr': lossTr,
			'lossVal': lossVal,
			'iouTr':mIOU_tr,
			'iouVal':mIOU_val,
			}, args.save_dir+ '/checkpoint.pth.tar')

		#save model also
		# if overall_acc_val > best_model_acc:
		# 	best_model_acc = overall_acc_val
		model_file_name=args.save_dir+'/model_'+ str(epoch+1) +'.pth'
		torch.save(model.state_dict(), model_file_name)
		with open('../acc/acc_'+ str(epoch)+'.txt','w') as log:
		    log.write("\nEpoch: %d\t Overall Acc (Tr): %.4f\t Overall Acc (Val): %.4f\t mIOU (Tr): %.4f\t mIOU (Val): %.4f" % (epoch, overall_acc_tr, overall_acc_val, mIOU_tr, mIOU_val))
		    log.write('\n')
		    log.write('Per Class Training Acc: ' + str(per_class_acc_tr))
		    log.write('\n')
		    log.write('Per Class Validation Acc: ' + str(per_class_acc_val))
		    log.write('\n')
		    log.write('Per Class Training mIOU: ' + str(per_class_iu_tr))
		    log.write('\n')
		    log.write('Per Class Validation mIOU: ' + str(per_class_iu_val))

		logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.6f" % (epoch, lossTr, lossVal, overall_acc_tr, overall_acc_val,mIOU_tr, mIOU_val, lr))
		logger.flush()
		print("Epoch : " + str(epoch) + ' Details')
		print("\nEpoch No.: %d\tTrain Loss = %.4f\tVal Loss = %.4f\t Train acc = %.4f\t Val acc = %.4f\t mIOU(tr) = %.4f\t mIOU(val) = %.4f" % (epoch, lossTr, lossVal,overall_acc_tr, overall_acc_val, mIOU_tr, mIOU_val))

	logger.close()

if __name__ == '__main__':
	parser=ArgumentParser(description="Training UNet.")
	parser.add_argument('--data_dir',default="../data")
	parser.add_argument('--inwidth', type=int,default=224,help='width of the input patch')
	parser.add_argument('--inheight', type=int,default=224,help='heigth of the input patch')
	parser.add_argument('--max_epochs', type=int,default=100)
	parser.add_argument('--step_loss', type=int,default=30)
	parser.add_argument('--lr', type=float,default=0.01)
	parser.add_argument('--batch_size', type=int,default=16)
	parser.add_argument('--batch_size_val', type=int,default=4)
	parser.add_argument('--num_workers', type=int,default=4)
	parser.add_argument('--save_dir',default='../result')
	parser.add_argument('--visNet', type=bool,default=True)
	parser.add_argument('--resume', type=bool,default=False)
	parser.add_argument('--resumeLoc', default='../result/checkpoint.pth.tar')
	parser.add_argument('--classes', type=int, default=11, help='Number of classes in the dataset')
	parser.add_argument('--attrClasses', type=int, default=1, help='Number of attribute classes')
	parser.add_argument('--cached_data_file', default='unet_cache.p')
	parser.add_argument('--logFile', default='trainValLog.txt')
	parser.add_argument('--onGPU', default=True)
	parser.add_argument('--scaleIn', type=int, default=1, help='scaling factor for training the models at ')

	args=parser.parse_args()
	print(args)
	trainValSegmentation(args)