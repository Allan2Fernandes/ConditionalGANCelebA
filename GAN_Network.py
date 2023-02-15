import math
import random

import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, Reshape, MaxPool2D, Dense, LeakyReLU, Flatten, Conv2DTranspose, Lambda
from keras.activations import selu, tanh
import matplotlib.pyplot as plt


class GAN_Network:
    def __init__(self, image_shape, noise_dimensions, num_classes):
        self.image_shape = image_shape
        self.image_dim = image_shape[1]
        self.noise_dimensions = noise_dimensions
        self.num_classes = num_classes
        pass

    def build_Generator(self, ):
        filters = 128
        kernel_size = (4,4)
        padding = 'same'
        kernel_initializer = 'he_normal'


        # #Downsample it
        # input_layer = Input(shape=[noise_dimensions+num_classes])
        # x = Dense(units = (self.image_dim**2)*3)(input_layer) #This needs to be adjusted if not using square inputs
        # x = Reshape(target_shape=(self.image_dim, self.image_dim, 3))(x)
        # x = Conv2D(filters = filters, strides=2, padding=padding, kernel_size=kernel_size, use_bias=False, kernel_initializer=kernel_initializer)(x)
        # x = BatchNormalization()(x)
        # x = LeakyReLU()(x)
        # x = Conv2D(filters=filters*2, strides=2, padding=padding, kernel_size=kernel_size, use_bias=False, kernel_initializer=kernel_initializer)(x)
        # x = BatchNormalization()(x)
        # x = LeakyReLU()(x)
        # x = Conv2D(filters = filters*4, strides=2, padding=padding, kernel_size=kernel_size, use_bias=False, kernel_initializer=kernel_initializer)(x)
        # x = BatchNormalization()(x)
        # x = LeakyReLU()(x)
        #
        # #Bottleneck layer
        # x = Conv2D(filters=filters*8, strides=2, padding=padding, kernel_size=kernel_size, use_bias=False, kernel_initializer=kernel_initializer)(x)
        # x = BatchNormalization()(x)
        # x = LeakyReLU()(x)
        #
        # #Upsample it
        # x = Conv2DTranspose(filters=filters*4, strides=2, padding=padding, kernel_size=kernel_size, use_bias=False)(x)
        # x = BatchNormalization()(x)
        # x = Lambda(lambda x: selu(x=x), input_shape=x.shape)(x)
        #
        # x = Conv2DTranspose(filters=filters * 2, strides=2, padding=padding, kernel_size=kernel_size, use_bias=False)(x)
        # x = BatchNormalization()(x)
        # x = Lambda(lambda x: selu(x=x), input_shape=x.shape)(x)
        #
        # x = Conv2DTranspose(filters=filters, strides=2, padding=padding, kernel_size=kernel_size, use_bias=False)(x)
        # x = BatchNormalization()(x)
        # x = Lambda(lambda x: selu(x=x), input_shape=x.shape)(x)
        #
        # x = Conv2DTranspose(filters=3, strides=2, padding=padding, kernel_size=kernel_size, use_bias=False)(x)
        # x = Lambda(lambda x: tanh(x=x), input_shape=x.shape)(x)
        #
        # self.generator = tf.keras.Model(inputs = input_layer, outputs = x)
        # self.generator.summary()

        self.generator = tf.keras.models.Sequential([
            Dense(units=(self.image_dim * self.image_dim * 3), input_shape=[self.noise_dimensions+self.num_classes]),
            Reshape(target_shape=(self.image_dim, self.image_dim, 3)),
            # Downsample it to the bottleneck
            Conv2D(filters=filters, kernel_size=kernel_size, strides=2, padding='same', use_bias=False,
                   kernel_initializer='he_normal'),
            BatchNormalization(),
            LeakyReLU(),
            Conv2D(filters=filters * 2, kernel_size=kernel_size, strides=2, padding='same', use_bias=False,
                   kernel_initializer='he_normal'),
            BatchNormalization(),
            LeakyReLU(),
            Conv2D(filters=filters * 4, kernel_size=kernel_size, strides=2, padding='same', use_bias=False,
                   kernel_initializer='he_normal'),
            BatchNormalization(),
            LeakyReLU(),

            # Bottleneck layer
            Conv2D(filters=filters * 8, kernel_size=kernel_size, strides=2, padding='same',
                   kernel_initializer='he_normal', use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            # Upsample it to the input shape
            Conv2DTranspose(filters=filters * 4, activation='selu', strides=2, kernel_size=kernel_size, padding='same',
                            use_bias=False),
            BatchNormalization(),
            Conv2DTranspose(filters=filters * 2, activation='selu', strides=2, kernel_size=kernel_size, padding='same',
                            use_bias=False),
            BatchNormalization(),
            Conv2DTranspose(filters=filters, activation='selu', strides=2, kernel_size=kernel_size, padding='same',
                            use_bias=False),
            BatchNormalization(),
            Conv2DTranspose(filters=3, activation='tanh', strides=2, kernel_size=kernel_size,
                            padding='same', use_bias=False)
        ])
        self.generator.summary()

        pass

    def build_Discriminator(self):
        filters = 128
        kernel_size = (4, 4)
        padding = 'same'
        kernel_initializer = 'he_normal'
        self.image_shape = (self.image_shape[0], self.image_shape[1], self.image_shape[2] + self.num_classes)
        # input_layer = Input(shape=(self.image_shape))
        # x = Conv2D(filters = filters, strides = 2, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer, use_bias=False)(input_layer)
        # x = BatchNormalization()(x)
        # x = LeakyReLU()(x)
        #
        # x = Conv2D(filters=filters, strides=2, kernel_size=kernel_size, padding=padding, kernel_initializer=kernel_initializer, use_bias=False)(x)
        # x = BatchNormalization()(x)
        # x = LeakyReLU()(x)
        #
        # x = Conv2D(filters=filters*2, strides=2, kernel_size=kernel_size, padding=padding,kernel_initializer=kernel_initializer, use_bias=False)(x)
        # x = BatchNormalization()(x)
        # x = LeakyReLU()(x)
        #
        # x = Conv2D(filters=filters * 2, strides=2, kernel_size=kernel_size, padding=padding,kernel_initializer=kernel_initializer, use_bias=False)(x)
        # x = BatchNormalization()(x)
        # x = LeakyReLU()(x)
        #
        # x = Conv2D(filters=filters * 4, strides=2, kernel_size=kernel_size, padding=padding,kernel_initializer=kernel_initializer, use_bias=False)(x)
        # x = BatchNormalization()(x)
        # x = LeakyReLU()(x)
        #
        # # x = Conv2D(filters=filters * 4, strides=2, kernel_size=kernel_size, padding=padding,kernel_initializer=kernel_initializer, use_bias=False)(x)
        # # x = BatchNormalization()(x)
        # # x = LeakyReLU()(x)
        # x = Flatten()(x)
        # x = Dense(units=1, activation='sigmoid')(x)
        #
        # self.discriminator = tf.keras.Model(inputs= input_layer, outputs = x)
        # self.discriminator.summary()

        self.discriminator = tf.keras.models.Sequential([
            Input(shape=self.image_shape),

            Conv2D(filters=filters, strides=(2, 2), kernel_size=(4, 4), padding='same', kernel_initializer='he_normal',
                   use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Conv2D(filters=filters, strides=(2, 2), kernel_size=(4, 4), padding='same', kernel_initializer='he_normal',
                   use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Conv2D(filters=filters*2, strides=(2, 2), kernel_size=(4, 4), padding='same', kernel_initializer='he_normal',
                   use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Conv2D(filters=filters*2, strides=(2, 2), kernel_size=(4, 4), padding='same', kernel_initializer='he_normal',
                   use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Conv2D(filters=filters*4, strides=(2, 2), kernel_size=(4, 4), padding='same', kernel_initializer='he_normal',
                   use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Conv2D(filters=filters*4, strides=(2, 2), kernel_size=(4, 4), padding='same', kernel_initializer='he_normal',
                   use_bias=False),
            BatchNormalization(),
            LeakyReLU(),

            Flatten(),
            Dense(units=1, activation='sigmoid')
        ])
        self.discriminator.summary()
        pass

    def initialize_loss_function_and_optimizers(self):
        # self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        # self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        # self.loss_function = tf.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

        self.discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=.0001, clipvalue=1.0, decay=1e-8)
        self.generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate=.0001, clipvalue=1.0, decay=1e-8)

        self.loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

        pass

    def get_total_discriminator_loss(self, fake_image_classifications, real_image_classifications):
        #Create the fake image labels
        fake_image_labels = tf.zeros_like(fake_image_classifications)
        #Create the real image labels
        real_image_labels = tf.ones_like(real_image_classifications)

        #Calculate the fake loss
        fake_loss = self.loss_function(y_true=fake_image_labels, y_pred = fake_image_classifications)

        #Calculate the real loss
        real_loss = self.loss_function(y_true=real_image_labels, y_pred = real_image_classifications)

        #Concatenate for total loss
        total_loss = tf.concat([fake_loss, real_loss], axis = 0)
        return total_loss

    def get_total_generator_loss(self, generated_image_classifications):
        #Create labels for these classifications
        labels = tf.ones_like(generated_image_classifications)
        total_generator_loss = self.loss_function(y_true=labels, y_pred=generated_image_classifications)
        return total_generator_loss

    def create_test_image(self, step, epoch):
        class_string = ""
        #Get the one_hot_vector to concatenate with noise
        classification_label = [random.randrange(2)]
        if classification_label[0] == 0:
            class_string = "Female"
        else:
            class_string = "Male"
        one_hot_vector, _ = self.get_one_hot_tensors(classification_labels_batch=classification_label)
        #Create noise and generate a image using the generator
        noise = tf.random.normal(shape=[1, self.noise_dimensions])
        noise = tf.concat([noise, one_hot_vector], axis= -1)
        generated_tensor = self.generator(noise)
        test_image = generated_tensor[0]
        test_image = test_image*0.5 + 0.5

        #Display the image
        plt.imshow(test_image)
        plt.title("Epoch= {0} || Step = {1} || Class = {2}".format(epoch, step, class_string))
        plt.show()
        pass

    def get_one_hot_tensors(self, classification_labels_batch):
        # Get the one_hot vector for noise
        one_hot_vector = tf.one_hot(classification_labels_batch, depth=2)
        # Get the one_hot_filters
        one_hot_filters = tf.expand_dims(tf.expand_dims(one_hot_vector, axis=-1), axis=-1)
        one_hot_filters = tf.reshape(one_hot_filters,
                                     shape=(
                                         one_hot_filters.shape[0],
                                         1,
                                         1,
                                         one_hot_filters.shape[1]
                                     ))
        one_hot_filters = tf.tile(one_hot_filters, [1, 128, 128, 1])
        return one_hot_vector, one_hot_filters

    def train_model(self, images_dataset, classifications_dataset, epochs, continue_generator_training=None, continue_discriminator_training=None, starting_epoch=-1):
        if not starting_epoch == -1:
            self.generator = continue_generator_training
            self.discriminator = continue_discriminator_training
            pass
        for epoch in range(epochs):
            for step, (real_images, classifications_batch) in enumerate(zip(images_dataset, classifications_dataset)):
                with tf.GradientTape() as discriminator_tape, tf.GradientTape() as generator_tape:
                    # Get the batch size
                    batch_size = real_images.shape[0]
                    num_batches = 202599/batch_size
                    num_batches = math.floor(num_batches)
                    # TRAIN THE DISCRIMINATOR
                    one_hot_vector, one_hot_filters = self.get_one_hot_tensors(classification_labels_batch=classifications_batch)
                    # Create noise
                    noise = tf.random.normal(shape=[batch_size, self.noise_dimensions])
                    #Concatenate the noise and one_hot_vector
                    noise = tf.concat([noise, one_hot_vector], axis=-1)
                    # Create fake images from the noise
                    fake_images = self.generator(noise)
                    #Concatenate the fake images and one_hot_filters
                    fake_images = tf.concat([fake_images, one_hot_filters], axis = -1)
                    # Classify the fake images
                    fake_image_classifications = self.discriminator(fake_images)
                    #Concatenate the real images
                    real_images = tf.concat([real_images, one_hot_filters], axis=-1)
                    # Classify the real images
                    real_image_classifications = self.discriminator(real_images)
                    total_disc_loss = self.get_total_discriminator_loss(fake_image_classifications=fake_image_classifications,real_image_classifications=real_image_classifications)

                    # TRAIN THE GENERATOR
                    # Using the images, get the total generator loss
                    total_gen_loss = self.get_total_generator_loss(generated_image_classifications=fake_image_classifications)

                    pass

                # Use the loss to calculate gradient
                discriminator_gradient = discriminator_tape.gradient(total_disc_loss,self.discriminator.trainable_variables)
                # Using the gradient, do gradient descent
                self.discriminator_optimizer.apply_gradients(zip(discriminator_gradient, self.discriminator.trainable_variables))

                # Calculate gradient using the loss
                generator_gradient = generator_tape.gradient(total_gen_loss, self.generator.trainable_variables)
                # Use the gradients to adjust the weights
                self.generator_optimizer.apply_gradients(zip(generator_gradient, self.generator.trainable_variables))

                print("Step: {0}/{1} || Discriminator loss = {2} || Generator loss = {3}".format(step, num_batches, tf.reduce_sum(total_disc_loss), tf.reduce_sum(total_gen_loss)))
                if step%100 == 0:
                    self.create_test_image(step=step, epoch=epoch)
                    if step%600==0:
                        self.generator.save(f"Models/Generator_Epoch={epoch+starting_epoch}_Step={step}")
                        self.discriminator.save(f"Models/Discriminator_Epoch={epoch+starting_epoch}_Step={step}")
                pass #per batch in dataset
            self.generator.save(f"Models/Generator_Epoch={epoch+starting_epoch+1}_Step=LastStep")
            self.discriminator.save(f"Models/Discriminator_Epoch={epoch+starting_epoch+1}_Step=LastStep")
            pass #per epoch (whole dataset)
        pass #End of method




