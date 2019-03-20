import os
import cv2
import pickle
import numpy as np
import torch.utils.data

class LoadData:
    def __init__(self,data_dir,classes,attrClasses,cached_data_file,normVal=1.10):
        self.data_dir=data_dir
        self.classes=classes
        self.attrClasses=attrClasses
        self.cached_data_file=cached_data_file
        self.normVal=normVal

        self.trainImList=list()
        self.valImList=list()
        self.trainAnnotList=list()
        self.valAnnotList=list()
        self.trainAttrCls=list()
        self.valAttrCls=list()

        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.zeros(3, dtype=np.float32)
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        self.attrWeights = np.ones(attrClasses, dtype=np.float32)

    def compute_class_weights(self,histogram):
        normhist=histogram/ np.sum(histogram)
        for i in range(self.classes):
            self.classWeights[i]=1/(np.log(self.normVal + normhist[i]))

    def compute_attr_weights(self,list):
        histogram,_ =np.histogram(list,self.attrClasses)
        normhist = histogram/ np.sum(histogram)
        for i in range(self.attrClasses):
            self.attrWeights[i] = 1/ (np.log(self.normVal + normhist[i]))

    def readFile(self,filename,trainstg=False):
        if trainstg == True:
            global_hist=np.zeros(self.classes,dtype=np.float32)
        no_files=0
        with open(self.data_dir +'/'+ filename,'r') as textfile:
            for line in textfile:
                line_arr=line.split(',')
                img_file=((self.data_dir).strip()+'/'+line_arr[0].strip()).strip()
                label_file=((self.data_dir).strip()+'/'+line_arr[1].strip()).strip()
                class_file=line_arr[2].strip()
                label_img=cv2.imread(label_file,0)
                unique_values=np.unique(label_img)
                max_val=max(unique_values)
                min_val=min(unique_values)

                if trainstg ==True:
                    hist=np.histogram(label_img,self.classes)
                    global_hist+=hist[0]
                    rgb_img=cv2.imread(img_file)
                    self.mean[0]+=np.mean(rgb_img[:,:,0])
                    self.mean[1]+=np.mean(rgb_img[:,:,1])
                    self.mean[2]+=np.mean(rgb_img[:,:,2])
                    self.std[0]+=np.std(rgb_img[:,:,0])
                    self.std[1]+=np.std(rgb_img[:,:,1])
                    self.std[2]+=np.std(rgb_img[:,:,2])

                    self.trainImList.append(img_file)
                    self.trainAnnotList.append(label_file)
                    self.trainAttrCls.append(class_file)
                else:
                    self.valImList.append(img_file)
                    self.valAnnotList.append(label_file)
                    self.valAttrCls.append(class_file)
                if max_val > (self.classes-1) or min_val<0:
                    print('error. problem with labels check: %s', label_file)
                no_files+=1

            if trainstg==True:
                self.mean /= no_files
                self.std /= no_files
                # scale the mean and std from [0, 255] to [0, 1]
                #self.mean /= 255.0
                #self.std /= 255.0
                self.compute_class_weights(global_hist)
                # self.compute_attr_weights(self.trainAttrCls)
                print(self.mean, no_files)
            return 0

    def processData(self):
        read_train=self.readFile('train.txt',True)
        read_val=self.readFile('val.txt')

        if read_train==0 and read_val==0:
            data_dict=dict()
            data_dict['trainIm']=self.trainImList
            data_dict['trainAnnot']=self.trainAnnotList
            data_dict['trainAttrCls']=self.trainAttrCls
            data_dict['valIm']=self.valImList
            data_dict['valAnnot']=self.valAnnotList
            data_dict['valAttrCls']=self.valAttrCls

            data_dict['mean'] = self.mean
            data_dict['std'] = self.std
            data_dict['classWeights'] = self.classWeights
            # data_dict['AttrClassWeights'] = self.attrWeights

            pickle.dump(data_dict, open(self.cached_data_file, "wb"))
            return data_dict
        return None


class MyDataset(torch.utils.data.Dataset):
    def __init__(self,imlist,labelist,transform=None):
        self.imlist = imlist
        self.labelist = labelist
        self.transform = transform

    def __len__(self):
        return len(self.imlist)

    def __getitem__(self,idx):
        image_name = self.imlist[idx]
        label_name = self.labelist[idx]
        image = cv2.imread(image_name)
        label = cv2.imread(label_name, 0)
        if self.transform:
            [image, label] = self.transform(image,label)
        return (image, label)