import tensorflow as tf
from models.CNN import *

import warnings
warnings.filterwarnings("ignore")

# Datasets
tf.app.flags.DEFINE_string('train_images_dir', 'C:\\_Files\\MyProjects\\CityClassification\\data\\train\\', 'Training images data directory.')
tf.app.flags.DEFINE_string('val_images_dir', 'C:\\_Files\\MyProjects\\CityClassification\\data\\validation\\', 'Validation images data directory.')
tf.app.flags.DEFINE_string('test_images_dir', 'C:\\_Files\\MyProjects\\CityClassification\\data\\test\\', 'Testing images data directory.')

tf.app.flags.DEFINE_string('inference_dir', 'C:\\_Files\\MyProjects\\CityClassification\\inference_CNN', 'Directory where to save inference result.')

tf.app.flags.DEFINE_boolean('train', False, 'whether to train the network')
tf.app.flags.DEFINE_integer('num_epochs', 10, 'epochs to train')
tf.app.flags.DEFINE_integer('train_batch_size', 8, 'number of elements in a training batch')
tf.app.flags.DEFINE_integer('val_batch_size', 8, 'number of elements in a validation batch')

tf.app.flags.DEFINE_integer('height_of_image', 100, 'Height of the images.')
tf.app.flags.DEFINE_integer('width_of_image', 100, 'Width of the images.')
tf.app.flags.DEFINE_integer('num_channels', 3, 'Number of the channels of the images.')
tf.app.flags.DEFINE_integer('num_classes', 2, 'Number of classes.')

tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate of the optimizer')
tf.app.flags.DEFINE_float('keep_prob', 0.33, 'Keeping probability with dropout regularization')

tf.app.flags.DEFINE_integer('display_step', 5, 'Number of steps we cycle through before displaying detailed progress.')
tf.app.flags.DEFINE_integer('validation_step', 5, 'Number of steps we cycle through before validating the model.')

tf.app.flags.DEFINE_string('base_dir', 'C:\\_Files\\MyProjects\\CityClassification\\results\\', 'Directory in which results will be stored.')
tf.app.flags.DEFINE_integer('checkpoint_step', 5, 'Number of steps we cycle through before saving checkpoint.')
tf.app.flags.DEFINE_integer('max_to_keep', 5, 'Number of checkpoint files to keep.')

tf.app.flags.DEFINE_integer('summary_step', 5, 'Number of steps we cycle through before saving summary.')

tf.app.flags.DEFINE_string('model_name', 'CNN', 'name of model')

FLAGS = tf.app.flags.FLAGS


def main(argv=None):
    model = CNN(
        train_images_dir=FLAGS.train_images_dir,
        val_images_dir=FLAGS.val_images_dir,
        test_images_dir=FLAGS.test_images_dir,
        inference_dir = FLAGS.inference_dir,
        num_epochs=FLAGS.num_epochs,
        train_batch_size=FLAGS.train_batch_size,
        val_batch_size=FLAGS.val_batch_size,
        height_of_image=FLAGS.height_of_image,
        width_of_image=FLAGS.width_of_image,
        num_channels=FLAGS.num_channels,
        num_classes=FLAGS.num_classes,
        learning_rate=FLAGS.learning_rate,
        base_dir=FLAGS.base_dir,
        max_to_keep=FLAGS.max_to_keep,
        model_name=FLAGS.model_name,
        keep_prob=FLAGS.keep_prob
    )

    model.create_network()
    model.initialize_network()

    if FLAGS.train:
        model.train_model(FLAGS.display_step, FLAGS.validation_step, FLAGS.checkpoint_step, FLAGS.summary_step)
    else:
        model.test_model()


if __name__ == "__main__":
    tf.app.run()