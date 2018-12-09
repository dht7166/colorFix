import keras
from keras.callbacks import ModelCheckpoint
from generator import Train_Discriminator,Train_GAN
from model import SharpGan
import numpy as np
from keras.callbacks import Callback
import os
import cv2

class Monitor(Callback):
    def __init__(self):
        self.logs = {'loss':1.0,
                     'discriminator_loss':1.0,
                     'out_generator_loss':1.0}

    def on_epoch_end(self, epoch, logs=None):
        self.logs = logs



batchsize= 8
steps = int(2800/batchsize)
# Create the model
model = SharpGan()
model.D.summary()
model.G.summary()
model.GAN.summary()

CURRENT_CYCLE = 1
# Try to load old weights
if CURRENT_CYCLE >1:
    print("LOADING OLD WEIGHTS")
    model.GAN.load_weights('change_model/GAN_trained_' +str(CURRENT_CYCLE-1)+'.h5')
    print('*************************************')


train_D_fake = Train_Discriminator(model.G,'ground_truth','simulated',batchsize,True)
train_D  = Train_Discriminator(model.G,'ground_truth','simulated',batchsize,False)
train_GAN =Train_GAN('ground_truth','simulated',batchsize)


# Monitor for Discriminator and GAN
monitor_D_real = Monitor()
monitor_D = Monitor()
monitor_GAN = Monitor()

# Threshold for training
if not os.path.exists('image'):
    os.mkdir('image')

image_folder = os.path.join('image',str(CURRENT_CYCLE))
if not os.path.exists(image_folder):
    os.mkdir(image_folder)

def train(nb_epoch):
    # Model Checkpoint
    ckpt_g = ModelCheckpoint('change_model/Generator_'+str(CURRENT_CYCLE)+'.h5',monitor = 'out_generator_loss',save_best_only=False)

    epoch_D = 1
    epoch_G = 1
    train_result = open('train_'+str(CURRENT_CYCLE)+'.csv','w',0)
    train_result.write('epoch,d_loss,SSIM+L1,g_loss\n')
    for i in range(nb_epoch+1):
        print('TRAINING EPOCH ' + str(i + 1)+'/'+str(nb_epoch))


        x,y = train_D_fake.get_validation(20)
        d_loss = model.D.evaluate(x,y,batchsize,verbose=0)

        x,y = train_GAN.get_validation(20)
        g_loss = model.GAN.evaluate(x,y,batchsize,verbose=0)
        train_result.write(str(i)+','+ str(d_loss)+','+str(g_loss[1])+','+str(g_loss[2])+'\n')

        if i!=0 and i%2 == 0:
            # Draw prediction after 2 epoches
            prediction = model.G.predict(x)
            image_folder_current_epoch = os.path.join(image_folder,'epoch_'+str(i))
            if not os.path.exists(image_folder_current_epoch):
                os.mkdir(image_folder_current_epoch)
            for i in range(20):
                folder = os.path.join(image_folder_current_epoch,str(i+1)+'.jpg')
                pred_image = prediction[i,:,:,:]
                pred_image = np.clip(pred_image*255,0,255)
                cv2.imwrite(folder,pred_image)
            # also save weight for that epoch
            model.GAN.save(os.path.join(image_folder_current_epoch,'saved_weights.h5'))



        if (i == nb_epoch):
            break


        print("Current D_loss",d_loss)
        print("Current G_loss",g_loss)


        # Pre-train the Discriminator
        print("Train the Discriminator")
        model.D.fit_generator(generator=train_D,
                              steps_per_epoch=steps,
                              epochs= epoch_D,
                              validation_data=train_D.get_validation(20),
                              callbacks=[monitor_D_real],
                              max_queue_size=5)
        print("Train the Discriminator on Fake images")
        model.D.fit_generator(generator=train_D_fake,
                              steps_per_epoch=steps,
                              epochs=epoch_D,
                              validation_data=train_D_fake.get_validation(20),
                              callbacks=[monitor_D],
                              max_queue_size=5)

        print("AFTER DISCRIMINATOR TRAIN")
        x, y = train_D_fake.get_validation(20)
        d_loss = model.D.evaluate(x, y, batchsize, verbose=0)
        print('Discriminator loss on fake images',d_loss)

        # Train the Generator through GAN
        print("train the Generator/GAN")
        model.GAN.fit_generator(generator=train_GAN,
                                steps_per_epoch=steps,
                                epochs=epoch_G,
                                validation_data=train_GAN.get_validation(20),
                                callbacks=[ckpt_g,monitor_GAN],
                                max_queue_size=5)
        model.GAN.save('change_model/GAN_trained_'+str(CURRENT_CYCLE)+'_'+'.h5')



train(20)