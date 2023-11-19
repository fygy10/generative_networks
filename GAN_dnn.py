import os
from keras import layers, callbacks
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#MNIST DATASET
#only assign the training dataset for purposes of the GAN
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

#CREATE THE DIRECTORY FOR SAVING RESULTS
directory = 'results'
if not os.path.exists(directory):
    os.mkdir(directory)

#GAN CLASS
class GAN():

    #INITIALIZE THE PARAMETERS, MODELS, AND COMPILING
    def __init__(self, input_shape = (28, 28, 1), rand_vector_shape = (100,), lr = 0.002, beta = 0.3):
        self.img_shape = input_shape
        self.input_size = rand_vector_shape
        self.opt = tf.keras.optimizers.legacy.Adam(lr, beta) #use of legacy optimizer

        self.generator = self.generator_model()
        self.generator.compile(loss = 'binary_crossentropy', optimizer = self.opt, metrics = ['accuracy'])

        self.discriminator  = self.discriminator_model()
        self.discriminator.compile(loss = 'binary_crossentropy', optimizer = self.opt, metrics = ['accuracy'])

        self.discriminator.trainable = False    #prevent discriminator weights from being updated during the training

        #outlines the forward pass from noise to final output
        input = tf.keras.Input(self.input_size)
        generated_img = self.generator(input)
        output = self.discriminator(generated_img)

        self.GAN = tf.keras.Model(input, output, name='GAN')
        self.GAN.compile(loss = 'binary_crossentropy', optimizer = self.opt, metrics = ['accuracy'])

    #DISCRIMINATOR CONSTRUCTED WITH ONLY DENSE LAYERS (NON-CONVOLUTIONAL)
    def discriminator_model(self):
        model = tf.keras.models.Sequential(name = 'Discriminator')
        model.add(layers.Flatten(input_shape =self.img_shape))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.2))  #added to improve performance
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    #GENERATOR CONSTRUCTED WITH ONLY DENSE LAYERS (NON-CONVOLUTIONAL)
    def generator_model(self):
        model = tf.keras.models.Sequential(name = 'Generator')
        model.add(layers.Dense(256, activation='relu', input_shape = self.input_size))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(1024))
        model.add(layers.LeakyReLU(alpha=0.01))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dense(np.prod(self.img_shape), activation='relu')) #1D tensor of output shape
        model.add(layers.Reshape(self.img_shape))   #2D tensor of output shape
        assert model.output_shape == (None, 28, 28, 1)
        return model

    #TRAIN THE MODEL
    def train(self, X_train, batch_size=128, epochs=2000, save_interval=200):
        half_batch = batch_size // 2
        y_pos_train_dis = np.ones((half_batch, 1))   #half fake image labels for discriminator 
        y_neg_train_dis = np.zeros((half_batch, 1))  #half real image labels for discriminator
        y_train_GAN = np.ones((batch_size, 1))   #use for generator training on real images

        for epoch in range(epochs):

            X_pos_train_dis = X_train[np.random.randint(0, X_train.shape[0], half_batch)] #real images
            X_pos_train_dis = np.expand_dims(X_pos_train_dis, axis=-1)  #add extra dimension for greyscale
            X_neg_train_dis = self.generator.predict(np.random.normal(0, 1, (half_batch, *self.input_size))) #generated images (Gaussian)); mean, std.dev, size(# and shape))
            
            #labels and images
            X_train_dis = np.concatenate((X_neg_train_dis, X_pos_train_dis), axis=0) #join along first axis
            y_train_dis = np.concatenate((y_neg_train_dis, y_pos_train_dis), axis=0)

            #shuffle labels and images and return
            indices = np.arange(X_train_dis.shape[0])
            np.random.shuffle(indices)
            X_train_dis = X_train_dis[indices]
            y_train_dis = y_train_dis[indices]


            #generator training data; concatenated after generation
            noise = np.random.normal(0, 1, (batch_size,) + self.input_size)

            #discriminator loss function updated after each batch
            d_loss = self.discriminator.train_on_batch(X_train_dis, y_train_dis)

            #generator loss function updated after each batch
            g_loss = self.GAN.train_on_batch(noise, y_train_GAN)

            #save an image after fixed number of iterations
            if epoch % save_interval == 0:
                # Print metrics
                print(f"Epoch {epoch}: [D loss: {d_loss[0]} - acc.: {d_loss[1]*100:.2f}] [G loss: {g_loss}]")

                # Save generated images
                self.save_imgs(epoch)

    #SAVE IMAGES AS THE RESULT
    def save_imgs(self, epoch):
        file_path = os.path.join(directory, f'epoch_{epoch}.png')
        r, c = 4, 4
        noise = np.random.normal(0, 1, (r * c,) + self.input_size)
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(file_path)
        plt.close()

#RUN THE MODEL
gan = GAN()
gan.train(x_train)