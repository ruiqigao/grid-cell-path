""" Data Generator """
import math
import numpy as np
from utils import generate_vel_list


class DataGenerator(object):
  """ Generate Monte Carlo samples and trajectories. """
  def __init__(self, config, num_grid=1000, min=0, max=1):
    self.num_grid = num_grid
    self.min, self.max = min, max
    self.grid_length = (self.max - self.min) / (self.num_grid - 1)
    self.config = config

  def generate(self, num_data, max_dx=3, min_dx=0, num_step=1, dtype=2, test=False, visualize=False):
    if dtype == 1:
      place_pair = self.generate_two_dim_multi_type1(num_data)
    elif dtype == 2:
      place_pair = self.generate_two_dim_multi_type2(num_data, max_dx, min_dx, num_step,
                                                     test=test, visualize=visualize)
    else:
      raise NotImplementedError

    return place_pair

  def generate_two_dim_multi_type1(self, num_data):
    """ Generate Monte Carlo samples (x, x_prime) for the basis expansion model. """
    theta = np.random.random(size=int(num_data * 1.5)) * 2 * np.pi - np.pi
    length = np.abs(np.random.normal(size=int(num_data * 1.5)) * self.config.sigma_data) / self.grid_length

    x = length * np.cos(theta)
    y = length * np.sin(theta)
    vel = np.concatenate((np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1)), axis=-1)

    mu_max = np.fmin(self.num_grid - 1, self.num_grid - 1 - vel)
    mu_min = np.fmax(0, -vel)
    select_idx = np.where((mu_max[:, 0] > mu_min[:, 0]) & (mu_max[:, 1] > mu_min[:, 1]))[0][:num_data]
    mu_max, mu_min, vel = mu_max[select_idx], mu_min[select_idx], vel[select_idx]
    assert len(vel) == num_data

    start = np.random.random(size=(num_data, 2)) * (mu_max - mu_min) + mu_min
    end = start + vel

    vel *= self.grid_length

    place_pair = {'before': start, 'after': end, 'vel': vel}
    return place_pair

  def generate_two_dim_multi_type2(self, num_data, max_dx, min_dx, num_step, test=False,
                                   visualize=False):
    """
    Generate Monte Carlo samples (x, x+dx) for the transformation model, and trajectories for testing
    path integration.
    """
    if not test:
      # Generate Monte Carlo samples (x, x+dx) for the transformation model.
      theta_id = np.random.choice(np.arange(self.config.num_theta), size=(num_data, num_step))
      theta = theta_id * 2 * math.pi / self.config.num_theta
      length = np.sqrt(np.random.random(size=(num_data, num_step))) * (max_dx - min_dx) + min_dx
      x = length * np.cos(theta)
      y = length * np.sin(theta)

      vel_seq = np.concatenate((np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1)), axis=-1)

      vel_seq_cumsum = np.cumsum(vel_seq, axis=1)
      mu_max = np.fmin(self.num_grid - 1, np.min(self.num_grid - 1 - vel_seq_cumsum, axis=1))
      mu_min = np.fmax(0, np.max(-vel_seq_cumsum, axis=1))
      start = np.random.random(size=(num_data, 2)) * (mu_max - mu_min) + mu_min
      start = np.expand_dims(start, axis=1)

      mu_seq = np.concatenate((start, start + vel_seq_cumsum), axis=1)
      vel = vel_seq * self.grid_length
    else:
      # Generate long path trajectories for testing path integration.
      velocity = generate_vel_list(max_dx, min_dx)
      num_vel = len(velocity)
      if visualize:
        mu_start = np.reshape([5, 5], newshape=(1, 1, 2))
        vel_pool = np.where((velocity[:, 0] >= -1) & (velocity[:, 1] >= -1))
        vel_idx = np.random.choice(vel_pool[0], size=[num_data * 50, num_step])

        vel_grid_cumsum = np.cumsum(np.take(velocity, vel_idx, axis=0), axis=1)
        mu_seq = np.concatenate((np.tile(mu_start, [num_data * 50, 1, 1]), vel_grid_cumsum + mu_start), axis=1)
        mu_seq_new, vel_idx_new = [], []
        for i in range(len(mu_seq)):
          mu_seq_sub = mu_seq[i]
          if len(np.unique(mu_seq_sub, axis=0)) == len(mu_seq_sub):
            mu_seq_new.append(mu_seq[i])
            vel_idx_new.append(vel_idx[i])
        mu_seq, vel_idx = np.stack(mu_seq_new, axis=0), np.stack(vel_idx_new, axis=0)
        mu_seq_rs = np.reshape(mu_seq, [-1, (num_step + 1) * 2])
        select_idx = np.where(np.sum(mu_seq_rs >= self.num_grid, axis=1) == 0)[0][:num_data]
        vel_idx = vel_idx[select_idx]
        mu_seq = mu_seq[select_idx]
        vel = np.take(velocity, vel_idx, axis=0) * self.grid_length
      else:
        vel_idx = np.random.choice(num_vel, size=[num_data * 100, num_step])
        vel_grid_cumsum = np.cumsum(np.take(velocity, vel_idx, axis=0), axis=1)
        mu_max = np.fmin(self.num_grid - 3, np.min(self.num_grid - 2 - vel_grid_cumsum, axis=1))
        mu_min = np.fmax(3, np.max(-vel_grid_cumsum + 2, axis=1))

        select_idx = np.where(np.sum(mu_max <= mu_min, axis=1) == 0)[0][:num_data]
        vel_idx, vel_grid_cumsum = vel_idx[select_idx], vel_grid_cumsum[select_idx]
        vel_grid = np.take(velocity, vel_idx, axis=0)
        mu_max, mu_min = mu_max[select_idx], mu_min[select_idx]
        mu_start = np.random.sample(size=[num_data, 2])
        mu_start = np.expand_dims(np.round(mu_start * (mu_max - mu_min) + mu_min - 0.5), axis=1)
        mu_seq = np.concatenate((mu_start, mu_start + vel_grid_cumsum), axis=1)
        vel = vel_grid * self.grid_length
    assert len(mu_seq) == num_data
    place_seq = {'seq': mu_seq, 'vel': vel}
    return place_seq