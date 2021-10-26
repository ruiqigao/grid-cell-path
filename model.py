""" Representation model of grid cells. """
import math
import numpy as np
import tensorflow as tf

from utils import construct_block_diagonal_weights


class GridCell(object):
  def __init__(self, config):
    self.beta1 = config.beta1
    self.place_dim = config.place_dim
    self.block_size = config.block_size
    self.num_grid = int(np.sqrt(self.place_dim))
    self.num_group = config.num_group
    self.grid_cell_dim = int(self.num_group * self.block_size)
    self.sigma = np.asarray(config.sigma, dtype=np.float32)
    assert self.num_group * self.block_size == self.grid_cell_dim
    assert self.num_grid * self.num_grid == self.place_dim
    self.num_theta = config.num_theta
    self.num_dtheta = config.num_dtheta

    self.max_dx = config.max_dx

    self.num_step = config.num_step
    self.lr = tf.placeholder(dtype=np.float32, name='lr')
    self.grid_length = config.place_size / (self.num_grid - 1)
    self.config = config

    # initialize v and u weights
    self.v = tf.get_variable('v', initializer=tf.convert_to_tensor(np.random.normal(
      scale=0.001, size=[self.num_grid, self.num_grid, self.grid_cell_dim]), dtype=tf.float32))
    self.u = []
    for i in range(len(self.sigma)):
      self.u.append(tf.get_variable('u' + str(i), initializer=tf.convert_to_tensor(np.random.normal(
        scale=0.001, size=[self.num_grid, self.num_grid, self.grid_cell_dim]), dtype=tf.float32)))

    # construct B weights
    self.B = construct_block_diagonal_weights(num_channel=config.num_theta, num_block=self.num_group,
                                              block_size=self.block_size, name='B', antisym=config.antisym, diag=False)
    self.T = tf.reshape(
      construct_block_diagonal_weights(num_channel=1, num_block=self.num_group, block_size=self.block_size, name='T',
                                       antisym=config.antisym, diag=False),
      [self.num_group, self.block_size, self.block_size])

    v_reshape = tf.reshape(self.v, [-1, self.grid_cell_dim])
    self.I = tf.matmul(v_reshape, v_reshape, transpose_a=True) / tf.matmul(tf.expand_dims(tf.sqrt(
      tf.reduce_sum(v_reshape ** 2, axis=0)), axis=-1),
      tf.expand_dims(tf.sqrt(tf.reduce_sum(v_reshape ** 2, axis=0)), axis=0))
    self.I2 = tf.matmul(v_reshape, v_reshape, transpose_b=True)

  def build_model(self):
    # compute kernel loss
    self.x = tf.placeholder(shape=[None, 2], dtype=tf.float32, name='x')
    self.x_prime = tf.placeholder(shape=[None, 2], dtype=tf.float32, name='x_prime')
    dx_square = tf.reduce_sum(((self.x - self.x_prime) * self.grid_length) ** 2, axis=1)

    v = self.get_grid_code(self.x)
    loss1 = []
    for i, sigma in enumerate(self.sigma):
      u = self.get_grid_code(self.x_prime, mode='decode', idx=i)
      kernel = tf.exp(- dx_square / self.sigma[i] / self.sigma[i] / 2.0)
      loss1.append(tf.reduce_mean((tf.reduce_sum(v * u, axis=1) - kernel) ** 2) * 30000)
    self.loss_kernel = tf.reduce_sum(tf.stack(loss1)) * self.config.weight_kernel

    # compute rotation loss of v
    self.place_seq2 = tf.placeholder(shape=[None, self.num_step + 1, 2], dtype=tf.float32, name='place_seq2')
    self.vel2 = tf.placeholder(shape=[None, self.num_step, 2], dtype=tf.float32, name='vel2')

    grid_code_seq2 = self.get_grid_code(self.place_seq2)
    grid_code = grid_code_seq2[:, 0]
    loss_transformation = tf.constant(0.0)
    for step in range(self.num_step):
      current_M = self.get_M(self.vel2[:, step])
      grid_code = self.motion_model(current_M, grid_code)
      loss_transformation = loss_transformation + tf.reduce_mean(
        tf.reduce_sum(tf.square(grid_code - grid_code_seq2[:, step + 1]), axis=-1)) * 30000

    self.loss_transformation = self.config.weight_transformation * loss_transformation / self.num_step

    # compute local isometry loss
    if self.config.weight_isometry == 0:
      self.loss_isometry = tf.constant(0.)
    else:
      theta_id1 = tf.random.uniform([self.config.num_data], maxval=self.num_theta, dtype=tf.int32)
      theta_id2 = tf.random.uniform([self.config.num_data], maxval=self.num_theta, dtype=tf.int32)
      B_theta1 = tf.gather(self.B, theta_id1)
      B_theta2 = tf.gather(self.B, theta_id2)
      v = self.get_grid_code(self.place_seq2[:self.config.num_data, 0])
      Bv1, Bv2 = self.motion_model_block(B_theta1, v), self.motion_model_block(B_theta2, v)
      Bv1_norm, Bv2_norm = tf.norm(Bv1, axis=-1), tf.norm(Bv2, axis=-1)
      self.loss_isometry = tf.reduce_mean((Bv1_norm - Bv2_norm) ** 2) * self.config.weight_isometry * 30000 * 16

    # compute regularization loss
    self.loss_reg_u = []
    for u in self.u:
      self.loss_reg_u.append(tf.reduce_sum(u ** 2))
    self.loss_reg_u = tf.reduce_sum(tf.stack(self.loss_reg_u)) * self.config.weight_reg_u

    # add up to total loss
    self.loss_total = self.loss_kernel + self.loss_transformation + self.loss_reg_u + self.loss_isometry

    optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1)
    trainable_vars = tf.trainable_variables()
    self.apply_grads = optim.minimize(self.loss_total, var_list=trainable_vars)

    # normalization and clip to positive operations
    v_gp = tf.reshape(self.v, [self.num_grid, self.num_grid, self.num_group, self.block_size])
    v_gp_norm = tf.nn.l2_normalize(v_gp, axis=-1) / np.sqrt(self.num_group)
    v_norm = tf.reshape(v_gp_norm, [self.num_grid, self.num_grid, self.grid_cell_dim])
    self.norm_grads = tf.assign(self.v, tf.stop_gradient(v_norm))

    self.clip_v = tf.assign(self.v, tf.stop_gradient(tf.clip_by_value(self.v, 0, np.inf)))
    self.clip_u = []
    for u in self.u:
      self.clip_u.append(tf.assign(u, tf.stop_gradient(tf.clip_by_value(u, 0, np.inf))))

  def get_grid_code(self, place_, mode='encode', idx=None):
    if mode == 'encode':
      weights = self.v
    elif mode == 'decode':
      assert idx is not None
      weights = self.u[idx]
    else:
      raise NotImplementedError
    grid_code = tf.contrib.resampler.resampler(tf.transpose(
      tf.expand_dims(weights, axis=0), perm=[0, 2, 1, 3]), tf.expand_dims(place_, axis=0))
    grid_code = tf.reshape(grid_code, tf.concat([tf.shape(place_)[:-1], tf.TensorShape(self.grid_cell_dim)], axis=0))
    return grid_code

  def dx_to_theta_id_dr(self, dx):
    assert len(dx._shape_as_list()) == 2
    theta = tf.math.atan2(dx[:, 1], dx[:, 0]) % (2 * math.pi)
    theta_id = tf.cast(tf.round(theta / (2 * math.pi / self.config.num_theta)), tf.int32)
    dr = tf.sqrt(tf.reduce_sum(dx ** 2, axis=-1))

    return theta_id, dr

  def get_M(self, vel_):
    def B(theta):
      return tf.gather(self.B, theta)

    vel = tf.reshape(vel_, [-1, 2])
    theta_id, dr = self.dx_to_theta_id_dr(vel)
    dr = tf.reshape(dr, [-1, 1, 1, 1])

    B_theta = B(theta_id)
    M = B_theta * dr + self.matmul_large(B_theta, B_theta) * (dr ** 2) / 2.

    return M + tf.reshape(tf.eye(self.block_size), [1, 1, self.block_size, self.block_size])

  def matmul_large(self, A, B):
    assert A.shape.as_list()[-1] == B.shape.as_list()[-2]
    batch_shape = tf.shape(A)[:-2]
    i, j, k = A.shape.as_list()[-2], A.shape.as_list()[-1], B.shape.as_list()[-1]
    A = tf.reshape(A, [-1, i, j])
    B = tf.reshape(B, [-1, j, k])
    n = tf.shape(A)[0]
    n_batch = tf.cast(tf.ceil(tf.cast(n, tf.float32) / np.float(self.config.fold)), tf.int32)

    if A.shape.as_list()[0] is not None and A.shape.as_list()[0] < 100:
      self.config.fold = 1
      n_batch = A.shape.as_list()[0]

    AB = []
    for tt in range(self.config.fold):
      slice = tf.range(tt * n_batch, tf.minimum((tt + 1) * n_batch, n))
      A_gp = tf.gather(A, slice, axis=0)
      B_gp = tf.gather(B, slice, axis=0)
      AB_gp = tf.matmul(A_gp, B_gp)
      AB.append(AB_gp)
    AB = tf.concat(AB, axis=0)
    AB = tf.reshape(AB, tf.concat([batch_shape, tf.TensorShape([i, k])], axis=0))

    return AB

  def motion_model(self, M, grid_code_):
    v = tf.reshape(grid_code_,
                   tf.concat([tf.shape(grid_code_)[:-1], tf.TensorShape([self.num_group, self.block_size, 1])], axis=0))

    grid_code = self.matmul_large(M, v)
    grid_code = tf.reshape(grid_code, tf.shape(grid_code_))

    return grid_code

  def motion_model_block(self, M, grid_code_):
    v = tf.reshape(grid_code_,
                   tf.concat([tf.shape(grid_code_)[:-1], tf.TensorShape([self.num_group, self.block_size, 1])], axis=0))

    grid_code = self.matmul_large(M, v)
    grid_code = tf.reshape(grid_code,
                           tf.concat([tf.shape(grid_code_)[:-1], tf.TensorShape([self.num_group, self.block_size])],
                                     axis=0))

    return grid_code

  def localization_model(self, u, grid_code_, grid_cell_dim, pd_pt=False, x_range=None, y_range=None, quantile=99.5):
    grid_code = tf.reshape(grid_code_, [-1, grid_cell_dim])
    u_reshape = tf.reshape(u, [-1, grid_cell_dim])
    place_code = tf.matmul(u_reshape, grid_code, transpose_b=True)
    place_pt_pd = None

    place_code = tf.transpose(
      tf.reshape(place_code, [self.num_grid, self.num_grid, -1]), perm=[2, 0, 1])
    if pd_pt:
      place_quantile = tf.contrib.distributions.percentile(place_code, quantile)
      place_pt_pool = tf.where(place_code - place_quantile >= 0)

      if x_range and y_range:
        choose_idx = tf.reshape(tf.where((place_pt_pool[:, 1] > x_range[0]) & (place_pt_pool[:, 1] < x_range[1]) & (
                  place_pt_pool[:, 2] > y_range[0]) & (place_pt_pool[:, 2] < y_range[1])), [-1])

        def fn1():
          return tf.gather(place_pt_pool, choose_idx)

        def fn2():
          place_quantile = tf.contrib.distributions.percentile(place_code, 97.5)
          place_pt_pool = tf.where(place_code - place_quantile >= 0)
          choose_idx = tf.reshape(tf.where(
            (place_pt_pool[:, 1] > x_range[0]) & (place_pt_pool[:, 1] < x_range[1]) & (
                    place_pt_pool[:, 2] > y_range[0]) & (place_pt_pool[:, 2] < y_range[1])), [-1])

          def fn3():
            return tf.gather(place_pt_pool, choose_idx)

          def fn4():
            place_quantile = tf.contrib.distributions.percentile(place_code, 94.5)
            place_pt_pool = tf.where(place_code - place_quantile >= 0)
            choose_idx = tf.reshape(tf.where(
              (place_pt_pool[:, 1] > x_range[0]) & (place_pt_pool[:, 1] < x_range[1]) & (
                      place_pt_pool[:, 2] > y_range[0]) & (place_pt_pool[:, 2] < y_range[1])), [-1])
            return tf.gather(place_pt_pool, choose_idx)

          place_pt_pool = tf.cond(tf.equal(tf.shape(choose_idx)[0], 0), fn4, fn3)
          return place_pt_pool

        place_pt_pool = tf.cond(tf.equal(tf.shape(choose_idx)[0], 0), fn2, fn1)

      place_pt_pd_x = tf.contrib.distributions.percentile(place_pt_pool[:, 1], 50.0)
      place_pt_pd_y = tf.contrib.distributions.percentile(place_pt_pool[:, 2], 50.0)
      place_pt_pd = tf.stack((place_pt_pd_x, place_pt_pd_y))
      place_pt_pd = tf.cast(place_pt_pd, tf.float32)

    return tf.squeeze(place_code), place_pt_pd

  def test_error_correction(self, noise_level=0.1, quantile=99.5):
    x = tf.random.uniform(shape=[2], minval=self.num_grid // 2 - self.num_grid // 4,
                          maxval=self.num_grid // 2 + self.num_grid // 4, dtype=tf.int32)
    x = tf.cast(x, tf.float32)
    v = self.get_grid_code(x)
    noise = tf.random.normal(tf.shape(v), mean=0, stddev=noise_level / np.sqrt(self.grid_cell_dim))
    v = v + noise

    if self.config.decode_weights == 'u':
      decode_weights = self.u[0]
    elif self.config.decode_weights == 'v':
      decode_weights = self.v
    else:
      raise NotImplementedError

    _, x_hat = self.localization_model(decode_weights, v, self.grid_cell_dim, pd_pt=True, quantile=quantile)
    return x, x_hat

  def path_integral(self, test_num, noise_type=None, noise_level=0):
    # build testing model
    self.place_init_test = tf.placeholder(shape=[2], dtype=tf.float32)
    self.vel2_test = tf.placeholder(shape=[test_num, 2], dtype=tf.float32)

    vu_heatmap_list = []
    vu_heatmap_block_list = []
    x_list = [[], []]

    grid_code = [self.get_grid_code(self.place_init_test), self.get_grid_code(self.place_init_test),
                 self.get_grid_code(self.place_init_test), self.get_grid_code(self.place_init_test)]

    vel = tf.reshape(self.vel2_test[0], [-1, 2])
    theta_id, _ = self.dx_to_theta_id_dr(vel)

    x_range, y_range = [[0, self.num_grid], [0, self.num_grid]], [[0, self.num_grid], [0, self.num_grid]]

    if self.config.decode_weights == 'u':
      decode_weights = self.u[0]
    elif self.config.decode_weights == 'v':
      decode_weights = self.v
    else:
      raise NotImplementedError

    for step in range(test_num + 1):
      if noise_type:
        if noise_type == "gaussian":
          sigma = noise_level / np.sqrt(self.grid_cell_dim)
          noise = tf.random.normal(tf.shape(grid_code[0]), mean=0, stddev=sigma)
          grid_code = [x + noise for x in grid_code]
        elif noise_type == "dropout":
          idx = tf.cast(tf.random_shuffle(np.arange(self.grid_cell_dim)), tf.float32)
          mask = tf.cast(idx < self.grid_cell_dim * (1 - noise_level), tf.float32)
          grid_code = [x * mask for x in grid_code]
        else:
          raise NotImplementedError
      for i in range(len(x_list)):
        vh_heatmap, x = self.localization_model(decode_weights, grid_code[i], self.grid_cell_dim, pd_pt=True,
                                                x_range=x_range[i], y_range=y_range[i])
        x_list[i].append(x)
        x_range[i] = [tf.cast(tf.maximum(x[0] - self.max_dx - 5, 0), tf.int64),
                      tf.cast(tf.minimum(x[0] + self.max_dx + 5, 80), tf.int64)]
        y_range[i] = [tf.cast(tf.maximum(x[1] - self.max_dx - 5, 0), tf.int64),
                      tf.cast(tf.minimum(x[1] + self.max_dx + 5, 80), tf.int64)]

        if i == 0:
          vu_heatmap_list.append(vh_heatmap)
      bk_list = []
      for gp in range(self.num_group):
        gp_id = slice(gp * self.block_size, (gp + 1) * self.block_size)
        v_times_u, _ = self.localization_model(
          tf.nn.l2_normalize(decode_weights[:, :, gp_id], axis=-1), tf.nn.l2_normalize(grid_code[0][gp_id]),
          self.block_size)
        bk_list.append(v_times_u)
      vu_heatmap_block_list.append(tf.stack(bk_list))

      if step < test_num:
        # if project_to_point is True and step > 0:

        # 0: vanilla, with [dr, theta] as input, without re-encoding
        M = self.get_M(self.vel2_test[step])
        grid_code[0] = self.motion_model(M, grid_code[0])

        # 1: [dr, theta] as input with re-encoding
        v = self.get_grid_code(x_list[1][-1])
        grid_code[1] = self.motion_model(M, v)

    self.integral_vu_heatmap, self.integral_vu_heatmap_block, self.integral_x_predict = \
      tf.stack(vu_heatmap_list), tf.stack(vu_heatmap_block_list), [tf.stack(x) for x in x_list]