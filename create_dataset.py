import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

# Create model and load weights


# Perform action on images
gt = 'ground_truth'
save_folder = 'simulated'
img_list = glob.glob(os.path.join(gt,'*.jpg'))
print(len(img_list))
for img in tqdm(img_list):
    image = cv2.imread(img)
    image = cv2.resize(image,(256,256))
    val = np.random.rand(1, 3)
    alpha = 0.3 + (2.0 - 0.3) * np.random.random()
    beta = -1.0 + (1.0 + 1.0) * np.random.random()
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta * 50)
    # n_alpha = 1/alpha
    # n_beta = -beta/alpha
    # image = cv2.convertScaleAbs(image,alpha = n_alpha,beta = n_beta*50)
    _, img_name = os.path.split(img)
    cv2.imwrite(os.path.join(save_folder, img_name), image)

