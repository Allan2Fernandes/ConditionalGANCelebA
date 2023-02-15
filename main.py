import DatasetBuilder
import matplotlib.pyplot as plt
import GAN_Network as Network
import tensorflow as tf

#Get the dataset
dataset_builder = DatasetBuilder.DatasetBuilder(batch_size=64)
images_dataset, classifications_dataset = dataset_builder.get_complete_dataset()
num_classes = 2
noise_dimensions = 128

#build the network
network = Network.GAN_Network((128,128,3), num_classes=num_classes, noise_dimensions=noise_dimensions)
#If training the network from scratch:
# network.build_Generator()
# network.build_Discriminator()

#Train the model
network.initialize_loss_function_and_optimizers()
network.train_model(
    images_dataset=images_dataset,
    classifications_dataset=classifications_dataset,
    epochs=10
    #continue_generator_training=tf.keras.models.load_model("Models/Generator_Epoch=3_Step=LastStep"),
    #continue_discriminator_training=tf.keras.models.load_model("Models/Discriminator_Epoch=3_Step=LastStep"),
    #starting_epoch = 3
)