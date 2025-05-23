from tensorflow import keras
from tensorflow.keras.utils import Sequence
import glob
import os
import cv2
import numpy as np
from random import shuffle
import tensorflow as tf # Added for tf.keras.Model type hint
from typing import Tuple, List # Added for Tuple and List type hints

def read_image(name: str, size: int, noise: bool = False) -> np.ndarray:
    """Reads, resizes, normalizes an image, and optionally adds noise."""
    img: np.ndarray = cv2.imread(name)
    img = cv2.resize(img,(size,size))
    img = img/255.0
    if noise:
        white_noise = np.random.random(img.shape)
        img = img + white_noise
        img = np.clip(img,0,1)
    return img

class Train_Discriminator(Sequence):
    """Keras Sequence to generate batches of data for training the Discriminator."""
    def __init__(self,model_G: tf.keras.Model, gt_folder: str, pred_folder: str, batch_size: int, fake: bool) -> None:
        """Initializes the sequence for discriminator training."""
        self.gt_folder: str = gt_folder
        self.pred_folder: str = pred_folder
        self.image_list: List[str] = glob.glob(os.path.join(gt_folder,'*.jpg'))
        self.length: int  = int(np.ceil(len(self.image_list) / batch_size))
        self.batch: int = batch_size
        self.G: tf.keras.Model = model_G
        self.fake: bool = fake

    def get_validation(self,size: int) -> tuple[np.ndarray, np.ndarray]:
        """Generates a batch of validation data."""
        x_batch: list = []
        y_batch: list = []
        fake_image: bool = self.fake
        # Ensure we don't go out of bounds if size is larger than image_list
        num_samples = min(size, len(self.image_list))
        for i in range(num_samples):
            _, img_name = os.path.split(self.image_list[i])
            if not fake_image:
                y_batch.append(1)
                x_batch.append(read_image(os.path.join(self.gt_folder, img_name),226))
            else:
                y_batch.append(0)
                image: np.ndarray = read_image(os.path.join(self.pred_folder, img_name),256)
                image = np.expand_dims(image, axis=0)
                image = self.G.predict(image)
                image = np.squeeze(image, axis=0)
                x_batch.append(image)
        return np.asarray(x_batch), np.asarray(y_batch).reshape((num_samples, 1, 1, 1))

    def on_epoch_end(self) -> None:
        """Called at the end of each epoch; can be used to shuffle data."""
        # Optionally shuffle self.image_list here if needed
        pass

    def __len__(self) -> int:
        """Returns the number of batches in the sequence."""
        return self.length

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Generates one batch of data."""
        x_batch: list = []
        y_batch: list = []
        fake_image: bool = self.fake
        
        start_idx = idx * self.batch
        end_idx = min((idx + 1) * self.batch, len(self.image_list))

        for i in range(start_idx, end_idx):
            _,img_name = os.path.split(self.image_list[i])
            if not fake_image:
                y_batch.append(1.0)
                x_batch.append(read_image(os.path.join(self.gt_folder,img_name),226,noise=True))
            else:
                y_batch.append(0.0)
                image: np.ndarray = read_image(os.path.join(self.pred_folder, img_name),256)
                image = np.expand_dims(image,axis = 0)
                image = self.G.predict(image)
                image = np.squeeze(image,axis=0)
                x_batch.append(image)
        return np.asarray(x_batch),np.asarray(y_batch).reshape((len(x_batch),1,1,1))


class Train_GAN(Sequence):
    """Keras Sequence to generate batches of data for training the combined GAN model."""
    def __init__(self,gt_folder: str, pred_folder: str, batch_size: int) -> None:
        """Initializes the sequence for GAN training."""
        self.image_list: List[str] = glob.glob(os.path.join(gt_folder,'*.jpg'))
        self.batch: int = batch_size
        self.gt_folder: str = gt_folder
        self.pred_folder: str = pred_folder
        self.length: int = int(np.ceil(len(self.image_list) / batch_size))

    def on_epoch_end(self) -> None:
        """Called at the end of each epoch; can be used to shuffle data."""
        # Optionally shuffle self.image_list here if needed
        pass

    def __len__(self) -> int:
        """Returns the number of batches in the sequence."""
        return self.length

    def __getitem__(self, idx: int) -> tuple[np.ndarray, list[np.ndarray]]: # Return type for GAN
        """Generates one batch of data for GAN training."""
        x_batch: list = []
        y_label_batch: list = [] # For the discriminator's label (all 1s for real)
        y_image_batch: list = [] # For the generator's target image
        
        start_idx = idx * self.batch
        end_idx = min((idx + 1) * self.batch, len(self.image_list))

        for i in range(start_idx, end_idx):
            _,img_name = os.path.split(self.image_list[i])
            x_batch.append(read_image(os.path.join(self.pred_folder,img_name),256))
            y_image_batch.append(read_image(os.path.join(self.gt_folder, img_name),226))
            y_label_batch.append(1.0)
        
        num_in_batch = len(x_batch)
        return np.asarray(x_batch),[np.asarray(y_image_batch),
                                    np.asarray(y_label_batch).reshape((num_in_batch,1,1,1))]

    def get_validation(self, size: int) -> tuple[np.ndarray, list[np.ndarray]]:
        """Generates a batch of validation data for GAN."""
        x_batch: list = []
        y_label_batch: list = []
        y_image_batch: list = []
        # Ensure we don't go out of bounds if size is larger than image_list
        num_samples = min(size, len(self.image_list))
        for i in range(num_samples):
            _,img_name = os.path.split(self.image_list[i])
            x_batch.append(read_image(os.path.join(self.pred_folder,img_name),256))
            y_image_batch.append(read_image(os.path.join(self.gt_folder,img_name),226))
            y_label_batch.append(1.0)

        return np.asarray(x_batch),[np.asarray(y_image_batch),
                                    np.asarray(y_label_batch).reshape((num_samples,1,1,1))]