import keras
from keras.callbacks import ModelCheckpoint
from generator import Train_Discriminator,Train_GAN
from model import SharpGan
import numpy as np
from keras.callbacks import Callback

class Monitor(Callback):
    def __init__(self,save_file):
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
print("LOADING OLD WEIGHTS")
model.GAN.load_weights('change_model/GAN_trained_2.h5')
print('*************************************')


train_D = Train_Discriminator(model.G,'ground_truth','simulated',batchsize)
train_GAN =Train_GAN('ground_truth','simulated',batchsize)


# Monitor for Discriminator and GAN
monitor_D = Monitor('Discriminator_train_double.txt')
monitor_GAN = Monitor('Gan_train_double.txt')

# Threshold for training
threshold = 4.0
stop_train_threshold = 0.01

def train(nb_epoch):
    # Model Checkpoint
    ckpt_gan = ModelCheckpoint('change_model/GAN_3.h5',monitor='loss',save_best_only=False)
    ckpt_g = ModelCheckpoint('change_model/Generator_3.h5',monitor = 'out_generator_loss',save_best_only=False)

    check_stop_early = {'discriminator':1.0,
                        'gan':1.0,
                        'generator':1.0,
                        'count':4}
    for i in range(nb_epoch):
        print('TRAINING EPOCH ' + str(i + 1)+'/'+str(nb_epoch))
        # Determine Training state using loss threshold
        d_loss = monitor_D.logs['loss']
        gan_loss = monitor_GAN.logs['discriminator_loss']
        g_loss = monitor_GAN.logs['out_generator_loss']

        # if (gan_loss<check_stop_early['gan'] and g_loss<check_stop_early['generator']):
        #     check_stop_early['count'] = 4
        #     check_stop_early['gan'] = gan_loss
        #     check_stop_early['generator'] = g_loss
        # else:
        #     check_stop_early['count'] = check_stop_early['count']-1
        # if check_stop_early['count']<0:
        #     print("STOPPING EARLY")
        #     break

        if (d_loss<stop_train_threshold and d_loss*threshold<gan_loss):
            global stop_train_threshold
            print("Discriminator too strong, not training this time")
            stop_train_threshold = stop_train_threshold/2
        else:
            # Pre-train the Discriminator
            print("Train the Discriminator")
            model.D.fit_generator(generator=train_D,
                                  steps_per_epoch=steps,
                                  epochs= 1,
                                  validation_data=train_D.get_validation(20),
                                  callbacks=[monitor_D],
                                  max_queue_size=5)

        # Train the Generator through GAN
        print("train the Generator/GAN")
        model.GAN.fit_generator(generator=train_GAN,
                                steps_per_epoch=steps,
                                epochs=4 if (gan_loss>threshold*d_loss) else 2,
                                validation_data=train_GAN.get_validation(20),
                                callbacks=[ckpt_g,ckpt_gan,monitor_GAN],
                                max_queue_size=5)
        model.GAN.save('change_model/GAN_trained_3.h5')


train(20)