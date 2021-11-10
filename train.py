""" Main training loop. """
import os
import time

from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from data import DataGenerator
from utils import draw_heatmap_2D, draw_path_to_target, draw_two_path, visualize, visualize_u


def train(config, model, sess, output_dir):
  model_dir = os.path.join(output_dir, 'model')

  if not tf.gfile.Exists(model_dir):
    tf.gfile.MakeDirs(model_dir)

  # build model
  model.build_model()
  data_generator = DataGenerator(config, max=config.place_size, num_grid=model.num_grid)

  # prepare data for path integral
  place_seq_test1 = data_generator.generate(1, max_dx=model.max_dx,
                                            num_step=config.integral_step, dtype=2, test=True, visualize=True)
  place_seq_test2 = data_generator.generate(100, max_dx=model.max_dx,
                                            num_step=config.integral_step, dtype=2, test=True)

  # initialize training
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())

  saver = tf.train.Saver(max_to_keep=10)

  if config.ckpt:
    saver.restore(sess, config.ckpt)

  # store graph in protobuf
  with open(model_dir + '/graph.proto', 'w') as f:
    f.write(str(tf.get_default_graph().as_graph_def()))

  # train
  start_time = time.time()

  stat_1_i = []
  stat_1 = {
    'loss_total': [],
    'loss_kernel': [],
    'loss_transformation': [],
    'loss_isometry': [],
    'loss_reg_u': []
  }

  for epoch in range(config.start_epoch, config.num_epochs):
    if epoch < 6000:
      lr_epoch = config.lr
    else:
      lr_epoch = config.lr2
    place_pair1 = data_generator.generate(config.num_data, dtype=1)
    place_seq2 = data_generator.generate(config.num_data, max_dx=model.max_dx,
                                         num_step=model.num_step, dtype=2)

    # update weights
    feed_dict = {model.x: place_pair1['before'],
                 model.x_prime: place_pair1['after'],
                 model.place_seq2: place_seq2['seq'],
                 model.vel2: place_seq2['vel'],
                 model.lr: lr_epoch}

    loss_total, loss_kernel, loss_transformation, loss_reg_u, loss_isometry, _ = sess.run(
      [model.loss_total, model.loss_kernel,
       model.loss_transformation, model.loss_reg_u,
       model.loss_isometry, model.apply_grads], feed_dict=feed_dict)

    # regularize weights
    if config.norm_weights:
      sess.run(model.norm_grads)
    if config.positive_u:
      sess.run(model.clip_u)
    if config.positive_v:
      sess.run(model.clip_v)

    if epoch % config.log_step == 0:
      end_time = time.time()
      u = sess.run(model.u)[0]
      log_info = '#{:s} Epoch #{:d}, loss_total: {:.4f}, loss_kernel: {:.4f}, loss_transformation: {:.4f}, loss_isometry: {:.4f}, loss_reg_u: {:.4f}, min_u: {:.4f}, max_u: {:.4f}, time: {:.2f}s, lr: {:.4f}' \
        .format(output_dir, epoch, loss_total, loss_kernel, loss_transformation, loss_isometry,
                loss_reg_u, u.min(), u.max(), end_time - start_time, lr_epoch)
      print(log_info)
      with open(os.path.join(output_dir, 'log.txt'), "a") as f:
        print(log_info, file=f)
      start_time = time.time()

      stat_1_i.append(epoch)
      stat_1['loss_total'].append(loss_total)
      stat_1['loss_kernel'].append(loss_kernel)
      stat_1['loss_transformation'].append(loss_transformation)
      stat_1['loss_isometry'].append(loss_isometry)
      stat_1['loss_reg_u'].append(loss_reg_u)

    syn_dir = os.path.join(output_dir, 'syn')
    syn_path_dir = os.path.join(output_dir, 'syn_path')
    if epoch == 0 or (epoch + 1) % config.log_step_large == 0 or epoch == config.num_epochs - 1:
      saver.save(sess, "%s/%s" % (model_dir, 'model.ckpt'), global_step=epoch)
      if not tf.gfile.Exists(syn_dir):
        tf.gfile.MakeDirs(syn_dir)
      feed_dict = dict()
      feed_dict.update({model.x: place_pair1['before'][:5],
                        model.x_prime: place_pair1['after'][:5],
                        model.place_seq2: place_seq2['seq'][:5]})
      feed_dict[model.vel2] = place_seq2['vel'][:5]
      I = sess.run(model.I)
      I2 = sess.run(model.I2)
      plt.figure()
      draw_heatmap_2D(I, cb=True)
      plt.savefig(os.path.join(syn_dir, 'I.png'))
      plt.close()
      plt.figure()
      draw_heatmap_2D(I2, cb=True)
      plt.savefig(os.path.join(syn_dir, 'I2.png'))
      plt.close()
      visualize(model, sess, syn_dir, epoch, final_epoch=True, result_dir='./tune_results')
      visualize_u(model, sess, syn_dir, epoch)

      p_i = 1
      p_n = len(stat_1)
      f = plt.figure(figsize=(20, p_n * 5))

      def plot_stats(stats, stats_i):
        nonlocal p_i
        for j, (k, v) in enumerate(stats.items()):
          plt.subplot(p_n, 1, p_i)
          plt.plot(stats_i, v)
          plt.ylabel(k)
          p_i += 1

      plot_stats(stat_1, stat_1_i)

      f.savefig(os.path.join(output_dir, 'stat.png'), bbox_inches='tight')
      plt.close(f)

      if (epoch + 1) % config.log_integral_step == 0:
        # test path integral
        test_path_integral(model, sess, place_seq_test1, visualize=True, test_dir=syn_path_dir, epoch=epoch)
        err = test_path_integral(model, sess, place_seq_test2)

        log_info = '%s %d epoch, path integral mse: %02f' % (output_dir, epoch, err)
        print(log_info)
        log_file = os.path.join(syn_path_dir, 'acc.txt')
        with open(log_file, 'a') as f:
          print(log_info, file=f)
        if epoch == config.num_epochs - 1:
          with open('integral_log.txt', 'a') as f:
            print(log_info, file=f)


def test_path_integral(model, sess, place_seq_test, visualize=False, test_dir=None, epoch=None):
  test_num = place_seq_test['seq'].shape[1] - 1
  err = []

  if visualize:
    assert test_dir is not None
    if not tf.gfile.Exists(test_dir):
      tf.gfile.MakeDirs(test_dir)
    if epoch is not None:
      test_dir = os.path.join(test_dir, str(epoch))
      tf.gfile.MakeDirs(test_dir)

  def visualize_heatmap(x_predict, vu_heatmap, vu_heatmap_block, file_name):
    num_gp_sqrt = max(int(np.ceil(np.sqrt(model.num_group))), 3)
    for j in range(test_num + 1):
      plt.figure(figsize=(num_gp_sqrt, num_gp_sqrt + 1))
      plt.subplot(num_gp_sqrt + 1, num_gp_sqrt, 1)
      draw_path_to_target(model.num_grid, place_seq_test['seq'][0, :j + 1])
      plt.subplot(num_gp_sqrt + 1, num_gp_sqrt, 2)
      draw_path_to_target(model.num_grid, x_predict[:j + 1])
      plt.subplot(num_gp_sqrt + 1, num_gp_sqrt, 3)
      draw_heatmap_2D(vu_heatmap[j])
      for gp in range(model.num_group):
        plt.subplot(num_gp_sqrt + 1, num_gp_sqrt, gp + num_gp_sqrt + 1)
        draw_heatmap_2D(vu_heatmap_block[j, gp], vmin=0, vmax=1)
      plt.tight_layout()
      plt.savefig(os.path.join(test_dir, file_name + '_' + str(j) + '.png'), bbox_inches='tight')
      plt.close()

  def visualize_path(x_gt, x_predict, file_name):
    plt.figure(figsize=(7, 7))
    draw_two_path(model.num_grid, x_gt, x_predict)
    plt.tight_layout()
    plt.savefig(os.path.join(test_dir, file_name + '.png'), bbox_inches='tight')
    plt.close()

  # path integral
  for i in range(len(place_seq_test['seq'])):
    feed_dict = {model.place_init_test: place_seq_test['seq'][i, 0],
                 model.vel2_test: place_seq_test['vel'][i]}
    x_predict, vu_heatmap, vu_heatmap_block = \
      sess.run([model.integral_x_predict, model.integral_vu_heatmap, model.integral_vu_heatmap_block],
               feed_dict=feed_dict)

    err.append([np.sqrt(np.sum((place_seq_test['seq'][i, 1:] - x[1:]) ** 2, axis=1)) for x in x_predict])

    if visualize:
      visualize_heatmap(x_predict[0], vu_heatmap, vu_heatmap_block, 'heatmap_%d' % i)
      visualize_path(place_seq_test['seq'][i, :test_num + 1], x_predict[0], 'theta_%d' % i)
      visualize_path(place_seq_test['seq'][i, :test_num + 1], x_predict[1], 'theta_re_encode_%d' % i)

  err = np.asarray(err)

  return err