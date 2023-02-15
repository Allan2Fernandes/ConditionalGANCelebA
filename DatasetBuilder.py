from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from PIL.Image import Image
import os
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import pandas as pd

class DatasetBuilder:
    def __init__(self, batch_size):
        #Get the images
        directory_path = "C:/Users/allan/Downloads/CelebA/img_align_celeba"
        self.target_size = (128, 128)
        dataset = tf.keras.utils.image_dataset_from_directory(
            directory=directory_path,
            labels=None,
            label_mode=None,
            class_names=None,
            color_mode='rgb',
            batch_size=None,
            image_size=self.target_size,
            shuffle=False,
            seed=None,
            validation_split=None,
            subset=None,
            interpolation='bilinear',
            follow_links=False,
            crop_to_aspect_ratio=False)

        self.images_dataset = dataset.map(map_func=self.map_function).batch(batch_size=batch_size, drop_remainder = True).prefetch(buffer_size = 1)

        #Get the attributes
        data_frame = pd.read_csv("C:/Users/allan/Downloads/CelebA/list_attr_celeba.csv")
        list_of_attributes = data_frame.columns
        attribute_name = 'Male'
        classifications = data_frame[attribute_name].to_numpy()
        classifications[classifications == -1] = 0
        classifications = tf.constant(classifications, tf.int32)
        #Create a dataset for the classifications
        classifications_dataset = tf.data.Dataset.from_tensor_slices(classifications)
        self.classifications_dataset = classifications_dataset.batch(batch_size=batch_size, drop_remainder=True).prefetch(1)




        pass

    def get_attribute_dataset(self, data_frame, attribute_name):
        #preprocess the classifications
        classifications = data_frame[attribute_name].to_numpy()
        classifications[classifications == -1] = 0
        classifications = tf.constant(classifications, tf.int32)
        #Get the one_hot vector for noise
        one_hot_vector = tf.one_hot(classifications, depth=2)
        #Get the one_hot_filters
        one_hot_filters = tf.expand_dims(tf.expand_dims(one_hot_vector, axis=-1), axis = -1)
        one_hot_filters = tf.reshape(one_hot_filters,
                                     shape=(
                                         one_hot_filters.shape[0],
                                         1,
                                         1,
                                         one_hot_filters.shape[1]
                                     ))
        one_hot_filters = tf.tile(one_hot_filters, [1, 128, 128, 1])
        print(one_hot_filters.shape )


    def map_function(self, datapoint):
        datapoint = (datapoint - 127.5) / 127.5
        return datapoint


    def get_complete_dataset(self):
        return self.images_dataset, self.classifications_dataset

    pass
