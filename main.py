""" Main function for training the representation model of grid cells. """
import argparse
import datetime
import os
import sys

import tensorflow as tf
import numpy as np

from data import DataGenerator
from model import GridCell
from train import train, test_path_integral
from utils import visualize, visualize_u, visualize_B


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()

# training parameters
parser.add_argument('--num_epochs', type=int, default=8000, help='Number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=0, help='Number of epochs to start')
parser.add_argument('--lr', type=float, default=0.003, help='Learning rate')  # TODO was 0.01
parser.add_argument('--lr2', type=float, default=0.0015, help='Learning rate')  # TODO was 0.01
parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 in Adam optimizer')
parser.add_argument('--gpu', type=str, default='1', help='Which gpu to use')

# simulated data parameters
parser.add_argument('--place_size', type=float, default=1.0, help='Length of the square place')
parser.add_argument('--max_dx', type=float, default=3, help='maximum  of velocity in rotation loss, measured in grids')
parser.add_argument('--fold', type=int, default=24, help='Number of simulated data at each iteration')
parser.add_argument('--num_data', type=int, default=90000, help='Number of simulated data at each iteration')
parser.add_argument('--sigma_data', type=float, default=0.48, help='Std of generated data pairs in kernel loss')

# model parameters
parser.add_argument('--num_theta', type=int, default=144, help='Number of discretized directions')
parser.add_argument('--sigma', metavar='N', type=float, nargs='+', default=[0.07], help='Std of Gaussian in A, support multi Gaussian. e.g. [0.05, 0.08, 0.12]')
parser.add_argument('--place_dim', type=int, default=1600, help='Dimensions of place, should be N^2')
parser.add_argument('--num_group', type=int, default=16, help='Number of blocks of grid cells')
parser.add_argument('--block_size', type=int, default=12, help='Size of each block')
parser.add_argument('--weight_transformation', type=float, default=0.5, help='Weight of rotation loss')
parser.add_argument('--weight_kernel', type=float, default=1.05, help='Weight of kernel loss')
parser.add_argument('--weight_isometry', type=float, default=0.5, help='Weight of local isometry loss, change to 0.0 if want to remove this loss')
parser.add_argument('--weight_reg_u', type=float, default=1.2, help='Weight of l2 regularization on u')
parser.add_argument('--num_step', type=int, default=1, help='Number of steps in rotation loss of training')
parser.add_argument('--norm_weights', type=boolean_string, default=True, help='True to normalize v')
parser.add_argument('--positive_u', type=boolean_string, default=True, help='True if want u to be positive')
parser.add_argument('--positive_v', type=boolean_string, default=False, help='True if want v to be positive')
parser.add_argument('--antisym', type=boolean_string, default=True, help='True if want v to be positive')
parser.add_argument('--decode_weights', type=str, default='v', help='u or v')

# error correction
parser.add_argument('--noise_type', type=str, default=None, help='None / gaussian / dropout')
parser.add_argument('--noise_level', type=float, default=1.0, help='Level of the noise')

# utils train (no need to change)
parser.add_argument('--log_file', type=str, default='test_log.txt', help='The output file for saving results')
parser.add_argument('--log_step', type=int, default=20, help='Number of log iterations')
parser.add_argument('--log_step_large', type=int, default=500, help='Number of larger log iterations')

# utils test
parser.add_argument('--mode', type=str, default='train', help='train / visualize / path_integration / error_correction')
parser.add_argument('--integral_step', type=int, default=5, help='Number of testing steps used in path integral')
parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint path to load')

FLAGS = parser.parse_args()


def main(_):
    if FLAGS.gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    tf.set_random_seed(1234)
    np.random.seed(0)

    model = GridCell(FLAGS)
    output_dir = os.path.join('output', os.path.splitext(os.path.basename(__file__))[0], datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    if len(sys.argv) > 1:
        output_dir = ''.join([output_dir] + sys.argv[1:])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'config.txt'), 'a') as f:
        print(FLAGS, file=f)

    def copy_source(file, output_dir):
        import shutil
        shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))
    copy_source(__file__, output_dir)

    with tf.Session() as sess:
        if FLAGS.mode == "train":  # training
            train(FLAGS, model, sess, output_dir)
        elif FLAGS.mode == "visualize":  # visualize weights
            # load model
            model.build_model()
            assert FLAGS.ckpt is not None, 'no checkpoint provided.'
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, FLAGS.ckpt)
            visualize(model, sess, output_dir)
            visualize_u(model, sess, output_dir)
            visualize_B(model, sess, output_dir)
        elif FLAGS.mode == "path_integration":  # perform path integration
            FLAGS.integral_step = max(30, FLAGS.integral_step)
            model.path_integral(FLAGS.integral_step, noise_type=FLAGS.noise_type, noise_level=FLAGS.noise_level)
            # load model
            assert FLAGS.ckpt is not None, 'no checkpoint provided.'
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, FLAGS.ckpt)
            print('Loading checkpoint {}.'.format(FLAGS.ckpt))

            test_dir = output_dir
            if not tf.gfile.Exists(test_dir):
                tf.gfile.MakeDirs(test_dir)
            data_generator_test = DataGenerator(FLAGS, max=FLAGS.place_size, num_grid=model.num_grid)

            place_seq_test = data_generator_test.generate(10, max_dx=FLAGS.max_dx,
                                                           num_step=FLAGS.integral_step, dtype=2, test=True, visualize=True)
            test_path_integral(model, sess, place_seq_test, visualize=True, test_dir=test_dir)

            place_seq_test2 = data_generator_test.generate(1000, max_dx=FLAGS.max_dx,
                                                      num_step=FLAGS.integral_step, dtype=2, test=True, visualize=False)
            err = test_path_integral(model, sess, place_seq_test2, visualize=False, test_dir=test_dir)
            np.save(os.path.join(test_dir, 'path_err.npy'), err)

            err = np.mean(err, axis=(0, 2))

            np.set_printoptions(threshold=int(1e8))
            print('%s err: %f %f' % (output_dir, err[0], err[1]))
            if FLAGS.log_file is not None:
                with open(FLAGS.log_file, "a") as f:
                    print('%s err: %f %f' % (output_dir, err[0], err[1]), file=f)
        elif FLAGS.mode == "error_correction":  # error correction
            x, x_hat = model.test_error_correction(noise_level=FLAGS.noise_level, quantile=99.5)
            assert FLAGS.ckpt is not None, 'no checkpoint provided.'
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, FLAGS.ckpt)
            print('Loading checkpoint {}.'.format(FLAGS.ckpt))

            x_value = np.zeros([1000, 2])
            x_hat_value = np.zeros([1000, 2])
            for i in range(1000):
                x_value_i, x_hat_value_i = sess.run([x, x_hat])
                x_value[i] = x_value_i
                x_hat_value[i] = x_hat_value_i
            err = np.sqrt(np.sum((x_value - x_hat_value) ** 2, axis=1)) / float(model.num_grid - 1)
            print('%s err: mean %f std %f' % (output_dir, np.mean(err), np.std(err)))
            if FLAGS.log_file is not None:
                with open(FLAGS.log_file, "a") as f:
                    print('%s err: mean %f std %f' % (output_dir, np.mean(err), np.std(err)), file=f)
        else:
            raise NotImplementedError


if __name__ == '__main__':
    tf.app.run()


