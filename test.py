
from typing import Callable, List, Tuple
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import atexit

from models import sesr, model_utils, adder

FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_integer('epochs', 300, 'Number of epochs to train')
tf.compat.v1.flags.DEFINE_integer('batch_size', 32, 'Batch size during training')
tf.compat.v1.flags.DEFINE_float('learning_rate', 2e-4, 'Learning rate for ADAM')
tf.compat.v1.flags.DEFINE_string('model_name', 'SESR', 'Name of the model')
tf.compat.v1.flags.DEFINE_bool('quant_W', False, 'Quantize weights')
tf.compat.v1.flags.DEFINE_bool('quant_A', False, 'Quantize activations')
tf.compat.v1.flags.DEFINE_bool('gen_tflite', False, 'Generate TFLITE')
tf.compat.v1.flags.DEFINE_integer('tflite_height', 1080, 'Height of LR image in TFLITE')
tf.compat.v1.flags.DEFINE_integer('tflite_width', 1920, 'Width of LR image in TFLITE')

import utils
import datetime

#Set some dataset processing parameters and some save/load paths
DATASET_NAME = 'div2k' if FLAGS.scale == 2 else 'div2k/bicubic_x4'
if not os.path.exists('logs/'):
  os.makedirs('logs/')
BASE_SAVE_DIR = 'logs/x2_models/' if FLAGS.scale == 2 else 'logs/x4_models/'
if not os.path.exists(BASE_SAVE_DIR):
  os.makedirs(BASE_SAVE_DIR)
print("BASE_SAVE_DIR: ", BASE_SAVE_DIR)



SUFFIX = 'QAT' if (FLAGS.quant_W and FLAGS.quant_A) else 'FP32'



##################################
##  EVALUATION  ##
##################################


def main(unused_argv):

    data_dir = os.getenv("TFDS_DATA_DIR", None)

    dataset_train, dataset_validation = tfds.load(DATASET_NAME, 
                                   split=['train', 'validation'], shuffle_files=True,
                                   data_dir=data_dir)
    dataset_train = dataset_train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset_validation = dataset_validation.prefetch(tf.data.experimental.AUTOTUNE)
    dataset_train = dataset_train.map(utils.rgb_to_y).cache()
    dataset_validation = dataset_validation.map(utils.rgb_to_y).cache()
    dataset_train = dataset_train.map(utils.patches).unbatch().shuffle(buffer_size=1000)

    dataset_validation = dataset_validation.map(utils.patches)

    #PSNR metric to be monitored while training.
    def psnr(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.image.psnr(y_true, y_pred, max_val=1.)

    def ssim(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.image.ssim(y_true, y_pred, max_val=1.)

    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # atexit.register(mirrored_strategy._extended._collective_ops._pool.close) # type: ignore (needed for tf2.7?)

    #Select the model to train.
    # with mirrored_strategy.scope():
    if FLAGS.model_name == 'SESR':
      if FLAGS.linear_block_type=='collapsed':
        LinearBlock_fn = model_utils.LinearBlock_c
      elif FLAGS.linear_block_type=='expanded':
        LinearBlock_fn = model_utils.LinearBlock_e
      elif FLAGS.linear_block_type=='collapsed_adder':
        LinearBlock_fn = model_utils.LinearBlock_c_adder
      else:
        raise "Please specify linear block type."
      
      print("linear_block_type: ", FLAGS.linear_block_type)

    #   model = sesr.SESR(
    #     m=FLAGS.m,
    #     feature_size=FLAGS.feature_size,
    #     LinearBlock_fn=LinearBlock_fn,
    #     quant_W=FLAGS.quant_W > 0,
    #     quant_A=FLAGS.quant_A > 0,
    #     gen_tflite = FLAGS.gen_tflite,
    #     mode='infer')


    model = tf.keras.models.load_model('/home/shawn/sesr/logs/adder_x2_models/SESR_m5_f16_x2_fs256_collapsed_adderTraining_FP32', \
        custom_objects={'psnr': psnr, 'ssim':ssim})



    result = model.evaluate(dataset_validation.batch(1))
    print(dict(zip(model.metrics_names, result)))


   

if __name__ == '__main__':
    tf.compat.v1.app.run()
