
import numpy as np
import cv2
import pickle
import torch.utils.data


class LoadData:
    '''
    This file for loading and caching the data and corresponding statistics
    Please note that this file needs one image and 2 labels (segmentation label and diagnostic label).
    '''
    def __init__(self, data_dir, classes, diagClasses, cached_data_file, normVal=1.10):
        self.data_dir = data_dir
        self.classes = classes
        self.classWeights = np.ones(self.classes, dtype=np.float32)
        self.normVal = normVal
        self.mean = np.zeros(3, dtype=np.float32)
        self.std = np.zeros(3, dtype=np.float32)
        self.trainImList = list()
        self.valImList = list()
        self.trainAnnotList = list()
        self.valAnnotList = list()
        self.diagClassTrain = list()
        self.diagClassVal = list()
        self.diagClasses = diagClasses
        self.diagWeights = np.ones(diagClasses, dtype=np.float32)

        self.cached_data_file = cached_data_file

    def compute_class_weights(self, histogram):
        normHist = histogram / np.sum(histogram)
        for i in range(self.classes):
            self.classWeights[i] = 1 / (np.log(self.normVal + normHist[i]))
            
    def compute_diag_weights(self, list):
        histogram, _ = np.histogram(list, self.diagClasses)
        normHist = histogram / np.sum(histogram)
        for i in range(self.diagClasses):
            self.diagWeights[i] = 1 / (np.log(self.normVal + normHist[i]))

    def readFile(self, fileName, trainStg=False):

        if trainStg == True:
            global_hist = np.zeros(self.classes, dtype=np.float32)

        no_files = 0
        with open(self.data_dir + '/' + fileName, 'r') as textFile:
            for line in textFile:
                #line = textFile.read()
                line_arr = line.split(',')
                img_file = ((self.data_dir).strip() + '/' + line_arr[0].strip()).strip()
                label_file = ((self.data_dir).strip() + '/' + line_arr[1].strip()).strip()
                class_file = int(line_arr[2].strip())

                label_img = cv2.imread(label_file, 0)

                unique_values = np.unique(label_img)
                max_val = max(unique_values)
                min_val = min(unique_values)

                if trainStg == True:
                    hist = np.histogram(label_img, self.classes)
                    global_hist += hist[0]

                    rgb_img = cv2.imread(img_file)

                    #rgb_img = rgb_img.transpose((2,0,1)) # convert from W x H X C to C X W X H
                    self.mean[0] += np.mean(rgb_img[:,:,0])
                    self.mean[1] += np.mean(rgb_img[:, :, 1])
                    self.mean[2] += np.mean(rgb_img[:, :, 2])

                    self.std[0] += np.std(rgb_img[:, :, 0])
                    self.std[1] += np.std(rgb_img[:, :, 1])
                    self.std[2] += np.std(rgb_img[:, :, 2])

                    self.trainImList.append(img_file)
                    self.trainAnnotList.append(label_file)
                    self.diagClassTrain.append(class_file)
                else:
                    self.valImList.append(img_file)
                    self.valAnnotList.append(label_file)
                    self.diagClassVal.append(class_file)

                if max_val > (self.classes - 1) or min_val < 0:
                    print('Some problem with labels. Please check.')
                    print('Label Image ID: ' + label_file)
                no_files += 1

        if trainStg == True:
            # divide the mean and std values by the sample space size
            self.mean /= no_files
            self.std /= no_files

            # scale the mean and std from [0, 255] to [0, 1]
            #self.mean /= 255.0
            #self.std /= 255.0

            #compute the class imbalance information
            self.compute_class_weights(global_hist)
            self.compute_diag_weights(self.diagClassTrain)
            print(self.mean, no_files)
        return 0

    def processData(self):
        print('Processing training data')
        return_val = self.readFile('train.txt', True)

        print('Processing validation data')
        return_val1 = self.readFile('val.txt')

        print('Pickling data')
        if return_val ==0 and return_val1 ==0:
            data_dict = dict()
            data_dict['trainIm'] = self.trainImList
            data_dict['trainAnnot'] = self.trainAnnotList
            data_dict['trainDiag'] = self.diagClassTrain
            data_dict['valIm'] = self.valImList
            data_dict['valAnnot'] = self.valAnnotList
            data_dict['valDiag'] = self.diagClassVal

            data_dict['mean'] = self.mean
            data_dict['std'] = self.std
            data_dict['classWeights'] = self.classWeights
            data_dict['diagClassWeights'] = self.diagWeights

            pickle.dump(data_dict, open(self.cached_data_file, "wb"))
            return data_dict
        return None




class MyDataset(torch.utils.data.Dataset):
    def __init__(self, imList, labelList, diagList, transform=None):
        self.imList = imList
        self.labelList = labelList
        self.diagList = diagList
        self.transform = transform

    def __len__(self):
        return len(self.imList)

    def __getitem__(self, idx):
        image_name = self.imList[idx]
        label_name = self.labelList[idx]
        image = cv2.imread(image_name)
        label = cv2.imread(label_name, 0)
        label2 = self.diagList[idx]
        if self.transform:
            [image, label, label2] = self.transform(image, label, label2)
        return (image, label, label2)
