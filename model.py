from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Input,Conv2D,LeakyReLU,BatchNormalization,MaxPool2D,Dense
from tensorflow.keras.layers import UpSampling2D,Add,Cropping2D,Concatenate,AveragePooling2D

W,H,C = (256,256,3)

class SharpGan():
    """Defines the SharpGAN model, including Generator, Discriminator, and combined GAN."""

    def loss_G(self,y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculates the generator loss, combining L1 distance and DSSIM."""
        # L1 Error, I am not using MSE
        L1_distance = K.mean(K.abs(y_true-y_pred))

        # SSIM error, or D-SSIM = (1-SSIM)
        dssim = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

        # TODO: Review SNR calculation logic and purpose.
        #Possibly add something here as well to penalize noise
        # SNR
        snr = K.sigmoid(K.var(y_pred-y_true)/K.var(y_true))
        # Assuming SNR was not part of the original loss based on the return statement provided in the previous content
        return self.l1_loss_weight* L1_distance + self.dssim_loss_weight*dssim

    def discriminator(self) -> tf.keras.Model:
        """Defines and returns the Discriminator model."""
        input = Input(shape=(226,226,C))
        # Layer 1 to 4 are all conv with downsampling
        x = input
        for i in range(1,7):
            filter = 16*(2**i)
            if filter>512:
                filter = 512
            x = Conv2D(filters=filter,kernel_size=(3,3),strides=(1,1),padding='valid',use_bias=False,
                       name = 'conv_'+str(i))(x)
            x = LeakyReLU(alpha=0.3)(x)
            x = BatchNormalization()(x)
            x = MaxPool2D(pool_size=(2,2))(x)


        output = Dense(1,activation='sigmoid',name='out_generator')(x)

        return Model(input,output,name='discriminator')


    def generator(self) -> tf.keras.Model:
        """Defines and returns the Generator model with U-Net like architecture."""
        input = Input((W,H,C))

        x = Conv2D(64,kernel_size=(3,3),padding='same',strides=(1,1),name=
                   'initial_conv',use_bias=False)(input)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
        skip_connection = []

        for i in range(1,5):
            skip_connection.append(x)
            filter = int(64*(2**i))
            x = Conv2D(filter,kernel_size=(3,3),padding='same',strides=(1,1),
                       name='conv_'+str(i),use_bias=False)(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = BatchNormalization()(x)
            x = AveragePooling2D(pool_size=(2,2))(x)


        skipping = x

        x = Conv2D(1024,kernel_size=(1,1),strides=(1,1),use_bias=False,name = 'conv_5')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)

        x = Concatenate()([skipping,x])

        x = Conv2D(1024,kernel_size=(1,1),strides=(1,1),use_bias=False,name = 'conv_6')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)

        shape = 32
        a_shape = 32
        for i in range(7,11):
            # Consider using Conv2dtranspose
            x = UpSampling2D(size=(2,2),name = 'upsample_'+str(i))(x)
            filters = int(512/(2**(i-6)))
            after_concat = int(1024/(2**(i-6)))

            cropped = Cropping2D(int((a_shape - shape) / 2))(skip_connection[6 - i])
            x = Concatenate(name='concat_data' + str(i))([x, cropped])
            x = Conv2D(filters= after_concat,kernel_size=(3,3),strides=(1,1),padding='valid',
                       use_bias=False,name = 'up_conv_'+str(i))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = BatchNormalization()(x)

            if after_concat!=filters:
                x = Conv2D(filters=filters,kernel_size=(1,1),name = 'padding_up_conv_'+str(i))(x)
                x = LeakyReLU(alpha=0.3)(x)
                x = BatchNormalization()(x)


            # Update shape
            shape =shape-2
            shape = shape*2
            shape = shape-(shape%2)
            a_shape = a_shape * 2

        for i in range(11,13):
            x = Conv2D(64,kernel_size=(1,1),name='final_out_conv_'+str(i))(x)
            x = LeakyReLU(alpha=0.3)(x)
            if i==12:
                continue
            x = BatchNormalization()(x)

        x = Conv2D(3,kernel_size=(1,1),name='out_generator',activation='sigmoid')(x)



        return Model(input,x,name='generator')


    def create_GAN(self) -> tf.keras.Model:
        """Creates and compiles the combined GAN model."""
        self.D.trainable = False
        output = self.D(self.G.output)
        self.GAN = Model(self.G.input,[self.G.output, output])
        GAN_optimizer = tensorflow.keras.optimizers.Adam(learning_rate=self.g_lr)
        self.GAN.compile(optimizer=GAN_optimizer, loss={
            'out_generator': self.loss_G,
            'discriminator': 'mean_squared_error'})
        return self.GAN


    def __init__(self, l1_loss_weight: float = 0.5, dssim_loss_weight: float = 0.1, d_lr: float = 1e-3, g_lr: float = 2e-3) -> None:
        """Initializes the SharpGan model, its components (D, G, GAN), and compiles them."""
        self.l1_loss_weight = l1_loss_weight
        self.dssim_loss_weight = dssim_loss_weight
        self.d_lr = d_lr
        self.g_lr = g_lr

        self.D = self.discriminator()
        D_optimizer = tensorflow.keras.optimizers.Adam(learning_rate=self.d_lr) #Hopefully helps when discriminator too strong
        self.D.compile(optimizer=D_optimizer, loss='mean_squared_error')

        self.G = self.generator()
        G_optimizer = tensorflow.keras.optimizers.Adam(learning_rate=self.g_lr)
        self.G.compile(optimizer=G_optimizer,loss = self.loss_G)

        self.GAN = self.create_GAN()

if __name__ == "__main__":
    gan_model = SharpGan()
    gan_model.G.summary()
    gan_model.D.summary()
    gan_model.GAN.summary()