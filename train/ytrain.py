import yloadData as ld
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
import yTransforms as myTransforms
# import DataSet as myDataLoader
import time
from argparse import ArgumentParser
from IOUEval import iouEval
import numpy as np
import r18unet
import ynetmobile
from torch.nn.functional import interpolate

def val(args,val_loader,model,criteria,criterion1):
	model.eval()
	iouEvalVal=iouEval(args.classes)
	iouDiagEvalVal = iouEval(args.attrClasses)
	class_loss = []
	epoch_loss=[]
	total_batches=len(val_loader)
	for i, (input,target, target2) in enumerate(val_loader):
		start_time=time.time()
		if args.onGPU==True:
			input = input.cuda()
			target=target.cuda()
			target2=target2.cuda()
		input_var=torch.autograd.Variable(input,volatile=True)
		target_var=torch.autograd.Variable(target,volatile=True)
		target2_var=torch.autograd.Variable(target2)
		output, output1 =model(input_var)

		loss=criteria(output, target_var)
		loss1 = criterion1(output1, target2_var)
		
		epoch_loss.append(loss.item())
		class_loss.append(loss1.item())  #loss.data[0]
		time_taken=time.time()-start_time

		iouEvalVal.addBatch(output.max(1)[1].data, target_var.data)
		iouDiagEvalVal.addBatch(output1.max(1)[1].data, target2_var.data)
		# print('[%d/%d] loss: %.3f time:%.2f' % (i, total_batches, loss.item(), time_taken))

	average_epoch_loss_val=sum(epoch_loss)/ len(epoch_loss)
	average_epoch_class_loss=sum(class_loss)/ len(class_loss)
	overall_acc,per_class_acc,per_class_iu,mIOU=iouEvalVal.getMetric()
	overall_acc1,per_class_acc1,per_class_iu1,mIOU1=iouDiagEvalVal.getMetric()

	return average_epoch_loss_val,overall_acc,per_class_acc,per_class_iu,mIOU, average_epoch_class_loss, overall_acc1, per_class_acc1, per_class_iu1, mIOU1



def train(args,train_loader,model,criteria,criterion1,optimizer,epoch):
	model.train()
	iouEvalTrain=iouEval(args.classes)
	iouDiagEvalTrain = iouEval(args.attrClasses)
	epoch_loss=[]    
	class_loss = []
	total_batches=len(train_loader)
	# print(total_batches)
	for i, (input,target, target2) in enumerate(train_loader):
		# print("input: ", input.size()[:], target.size()[:], input.size(0))
		start_time=time.time()
		if args.onGPU==True:
			input = input.cuda()
			target=target.cuda()
			target2=target2.cuda()
		input_var=torch.autograd.Variable(input)
		target_var=torch.autograd.Variable(target)
		target2_var=torch.autograd.Variable(target2)
		output, output1=model(input_var)
		# print('==================')
		# print(output.shape)
		# print(target_var.shape)
		# print('==================')
		optimizer.zero_grad()
		loss=criteria(output, target_var)
		loss1 = criterion1(output1, target2_var)
		optimizer.zero_grad()        
		loss1.backward(retain_graph=True)# you need to keep the graph from classification branch so that it can be used
                                            # during the update from the segmentation branch
		loss.backward()
		optimizer.step()

		epoch_loss.append(loss.item())  #loss.data[0]
		class_loss.append(loss1.item())  #loss.data[0]
		time_taken=time.time()-start_time

		iouEvalTrain.addBatch(output.max(1)[1].data, target_var.data)        
		iouDiagEvalTrain.addBatch(output1.max(1)[1].data, target2_var.data)

		# print('[%d/%d] loss: %.3f time:%.2f' % (i, total_batches, loss.item(), time_taken))

	average_epoch_loss_train=sum(epoch_loss)/ len(epoch_loss)
	overall_acc,per_class_acc,per_class_iu,mIOU=iouEvalTrain.getMetric()
	average_epoch_class_loss = sum(class_loss) / len(class_loss)
	overall_acc1, per_class_acc1, per_class_iu1, mIOU1 = iouDiagEvalTrain.getMetric()

	return average_epoch_loss_train,overall_acc,per_class_acc,per_class_iu,mIOU,average_epoch_class_loss, overall_acc1, per_class_acc1, per_class_iu1, mIOU1


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
	model = ynetmobile.MobileUNet(args.classes,args.attrClasses)
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
		y,y1 = model.forward(x)
		print("after forward...")
		g = viz.make_dot(y)
		g1 = viz.make_dot(y1)
		g.render(args.save_dir + 'yseg.png', view=False)
		g.render(args.save_dir + 'ycls.png', view=False)
	model = torch.nn.DataParallel(model)
	n_param = sum([np.prod(param.size()) for param in model.parameters()])
	print('network parameters: ' + str(n_param))

	#define optimization criteria
	weight = torch.from_numpy(data['classWeights'])
	print(weight)
	if args.onGPU == True:
		weight = weight.cuda()
	criteria = CrossEntropyLoss2d(weight)
	criteria1 = torch.nn.CrossEntropyLoss()

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
		ld.MyDataset(data['trainIm'], data['trainAnnot'], data['trainDiag'],transform=trainDatasetNoZoom),
		batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
	trainLoaderWithZoom = torch.utils.data.DataLoader(
		ld.MyDataset(data['trainIm'], data['trainAnnot'], data['trainDiag'], transform=trainDatasetWithZoom),
		batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
	valLoader = torch.utils.data.DataLoader(
		ld.MyDataset(data['valIm'], data['valAnnot'],data['valDiag'], transform=valDataset),
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
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,40,90],gamma=0.1)
	best_model_acc=0
	for epoch in range(start_epoch, args.max_epochs):
		scheduler.step(epoch)
		lr=0
		for param_group in optimizer.param_groups:
			lr= param_group['lr']
		# train(args,trainLoaderWithZoom,model,criteria,optimizer,epoch)
		lossTr,overall_acc_tr,per_class_acc_tr,per_class_iu_tr,mIOU_tr,lossTr1,overall_acc_tr1,per_class_acc_tr1,per_class_iu_tr1,mIOU_tr1=train(args,trainLoaderNoZoom,model,criteria,criteria1,optimizer,epoch)
		# print(per_class_acc_tr,per_class_iu_tr)
		lossVal, overall_acc_val, per_class_acc_val, per_class_iu_val, mIOU_val,lossVal1, overall_acc_val1, per_class_acc_val1, per_class_iu_val1, mIOU_val1 = val(args, valLoader, model, criteria,criteria1)

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
		    log.write('Classification Results')
		    log.write("\nEpoch: %d\t Overall Acc (Tr): %.4f\t Overall Acc (Val): %.4f\t mIOU (Tr): %.4f\t mIOU (Val): %.4f" % (
		        epoch, overall_acc_tr1, overall_acc_val1, mIOU_tr1, mIOU_val1))
		    log.write('\n')
		    log.write('Per Class Training Acc: ' + str(per_class_acc_tr1))
		    log.write('\n')
		    log.write('Per Class Validation Acc: ' + str(per_class_acc_val1))
		    log.write('\n')
		    log.write('Per Class Training mIOU: ' + str(per_class_iu_tr1))
		    log.write('\n')
		    log.write('Per Class Validation mIOU: ' + str(per_class_iu_val1))

		logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.6f" % (epoch, lossTr, lossVal, overall_acc_tr, overall_acc_val,mIOU_tr, mIOU_val,lossTr1,overall_acc_tr1,lossVal1,overall_acc_val1, lr))
		logger.flush()
		print("Epoch : " + str(epoch) + ' Details')
		print("\nEpoch No.: %d\tTrLoss = %.4f\tValLoss = %.4f\t Tracc = %.4f\t Valacc = %.4f\t mIOU(tr) = %.4f\t mIOU(val) = %.4f\tTrclsLoss = %.4f\tValclsLoss = %.4f\t Trclsacc = %.4f\t Valclsacc = %.4f\t" % (epoch, lossTr, lossVal,overall_acc_tr, overall_acc_val, mIOU_tr, mIOU_val,lossTr1,lossVal1,overall_acc_tr1,overall_acc_val1))

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
	parser.add_argument('--attrClasses', type=int, default=2, help='Number of attribute classes')
	parser.add_argument('--cached_data_file', default='ynet_cache.p')
	parser.add_argument('--logFile', default='ytrainValLog.txt')
	parser.add_argument('--onGPU', default=True)
	parser.add_argument('--scaleIn', type=int, default=1, help='scaling factor for training the models at ')

	args=parser.parse_args()
	print(args)
	trainValSegmentation(args)