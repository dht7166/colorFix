import keras
from keras.utils import Sequence
import glob
import os
import cv2
import numpy as np
from random import shuffle

def read_image(name,size,noise = False):
    img = cv2.imread(name)
    img = cv2.resize(img,(size,size))
    img = img/255.0
    if noise:
        white_noise = np.random.random(img.shape)
        img = img + white_noise
        img = np.clip(img,0,1)
    return img

class Train_Discriminator(Sequence):
    def __init__(self,model_G,gt_folder,pred_folder,batch_size,fake):
        self.gt_folder = gt_folder
        self.pred_folder = pred_folder
        self.image_list = glob.glob(os.path.join(gt_folder,'*.jpg'))
        self.length  = int(len(self.image_list)/batch_size)
        self.batch = batch_size
        self.G = model_G
        self.fake = fake

    def get_validation(self,size):
        x_batch = []
        y_batch = []
        fake_image = self.fake
        for i in range(size):
            _, img = os.path.split(self.image_list[i])
            if not fake_image:
                y_batch.append(1)
                x_batch.append(read_image(os.path.join(self.gt_folder, img),226))
            else:
                y_batch.append(0)
                image = read_image(os.path.join(self.pred_folder, img),256)
                image = np.expand_dims(image, axis=0)
                image = self.G.predict(image)
                image = np.squeeze(image, axis=0)
                x_batch.append(image)
        return np.asarray(x_batch), np.asarray(y_batch).reshape((size, 1, 1, 1))

    def on_epoch_end(self):
        pass

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_batch = []
        y_batch = []
        fake_image = self.fake
        for i in range(idx*self.batch,(idx+1)*self.batch):
            _,img = os.path.split(self.image_list[i])
            if not fake_image:
                y_batch.append(1.0)
                x_batch.append(read_image(os.path.join(self.gt_folder,img),226,noise=True))
            else:
                y_batch.append(0.0)
                image = read_image(os.path.join(self.pred_folder, img),256)
                image = np.expand_dims(image,axis = 0)
                image = self.G.predict(image)
                image = np.squeeze(image,axis=0)
                x_batch.append(image)
        return np.asarray(x_batch),np.asarray(y_batch).reshape((self.batch,1,1,1))





class Train_GAN(Sequence):
    def __init__(self,gt_folder,pred_folder,batch_size):
        self.image_list = glob.glob(os.path.join(gt_folder,'*.jpg'))
        self.batch = batch_size
        self.gt_folder = gt_folder
        self.pred_folder = pred_folder
        self.length = int(len(self.image_list) / batch_size)

    def on_epoch_end(self):
        pass


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x_batch = []
        y_batch = []
        y_image_batch = []
        for i in range(idx * self.batch, (idx + 1) * self.batch):
            _,img = os.path.split(self.image_list[i])
            x_batch.append(read_image(os.path.join(self.pred_folder,img),256))
            y_image_batch.append(read_image(os.path.join(self.gt_folder, img),226))
            y_batch.append(1.0)
        return np.asarray(x_batch),[np.asarray(y_image_batch),
                                    np.asarray(y_batch).reshape((self.batch,1,1,1))]

    def get_validation(self, size):
        x_batch = []
        y_batch = []
        y_image_batch = []
        for i in range(size):
            _,img = os.path.split(self.image_list[i])
            x_batch.append(read_image(os.path.join(self.pred_folder,img),256))
            y_image_batch.append(read_image(os.path.join(self.gt_folder,img),226))
            y_batch.append(1.0)
        return np.asarray(x_batch),[np.asarray(y_image_batch),
                                    np.asarray(y_batch).reshape((size,1,1,1))]