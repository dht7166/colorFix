import keras
from keras.callbacks import ModelCheckpoint
from generator import Train_Discriminator,Train_GAN
from model import SharpGan
import numpy as np
from keras.callbacks import Callback

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
# Try to load old weights
# print("LOADING OLD WEIGHTS")
# model.GAN.load_weights('change_model/GAN_trained_3.h5')
# print('*************************************')


train_D_fake = Train_Discriminator(model.G,'ground_truth','simulated',batchsize,True)
train_D  = Train_Discriminator(model.G,'ground_truth','simulated',batchsize,False)
train_GAN =Train_GAN('ground_truth','simulated',batchsize)


# Monitor for Discriminator and GAN
monitor_D_real = Monitor()
monitor_D = Monitor()
monitor_GAN = Monitor()

# Threshold for training
threshold = 1.0


def train(nb_epoch):
    # Model Checkpoint
    ckpt_g = ModelCheckpoint('change_model/Generator_1.h5',monitor = 'out_generator_loss',save_best_only=False)


    epoch_D = 1
    epoch_G = 2
    for i in range(nb_epoch):
        print('TRAINING EPOCH ' + str(i + 1)+'/'+str(nb_epoch))

        x,y = train_D.get_validation(20)
        d_loss = model.D.evaluate(x,y,batchsize,verbose=0)
        x,y = train_D_fake.get_validation(20)
        d_loss = (d_loss + model.D.evaluate(x,y,batchsize,verbose=0))/2

        x,y = train_GAN.get_validation(20)
        g_loss = model.GAN.evaluate(x,y,batchsize,verbose=0)

        print("Current D_loss",d_loss)
        print("Current G_loss",g_loss)

        if (d_loss*threshold>g_loss[2]):
            # Pre-train the Discriminator
            print("Train the Discriminator on Given images")
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
        else:
            # Train the Generator through GAN
            print("train the Generator/GAN")
            model.GAN.fit_generator(generator=train_GAN,
                                    steps_per_epoch=steps,
                                    epochs=epoch_G,
                                    validation_data=train_GAN.get_validation(20),
                                    callbacks=[ckpt_g,monitor_GAN],
                                    max_queue_size=5)
        model.GAN.save('change_model/GAN_trained_1.h5')



train(20)