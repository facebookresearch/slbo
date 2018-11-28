# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import List, Callable
import tensorflow as tf
import numpy as np
import lunzi.nn as nn
from lunzi import Tensor
from lunzi.Logger import logger
from slbo.utils.dataset import Dataset
from slbo.policies import BaseNNPolicy
from slbo.v_function import BaseVFunction


def average_l2_norm(x):
    return np.sqrt((x**2).mean())


# for damping, modify func_Ax
def conj_grad(mat_mul_vec: Callable[[np.ndarray], np.ndarray], b, n_iters=10, residual_tol=1e-10, verbose=False):
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    r_dot_r = r.dot(r)

    for i in range(n_iters):
        if verbose:
            logger.info('[CG] iters = %d, |Res| = %.6f, |x| = %.6f', i, r_dot_r, np.linalg.norm(x))
        z = mat_mul_vec(p)
        v = r_dot_r / p.dot(z)
        x += v * p
        r -= v * z
        new_r_dot_r = r.dot(r)
        if new_r_dot_r < residual_tol:
            break
        mu = new_r_dot_r / r_dot_r
        p = r + mu * p
        r_dot_r = new_r_dot_r
    return x


class TRPO(nn.Module):
    def __init__(self, dim_state: int, dim_action: int, policy: BaseNNPolicy, vfn: BaseVFunction, max_kl: float,
                 n_cg_iters: int, ent_coef=0.0, cg_damping=0.01, vf_lr=3e-4, n_vf_iters=3):
        super().__init__()
        self.dim_state = dim_state
        self.policy = policy
        self.ent_coef = ent_coef
        self.vf = vfn
        self.n_cg_iters = n_cg_iters
        self.max_kl = max_kl
        self.cg_damping = cg_damping
        self.n_vf_iters = n_vf_iters
        self.vf_lr = vf_lr

        # doing backtrace, so don't need to separate.
        self.flatten = nn.FlatParam(self.policy.parameters())
        self.old_policy: nn.Module = policy.clone()

        with self.scope:
            self.op_returns = tf.placeholder(dtype=tf.float32, shape=[None], name='returns')
            self.op_advantages = tf.placeholder(dtype=tf.float32, shape=[None], name='advantages')
            self.op_states = tf.placeholder(dtype=tf.float32, shape=[None, dim_state], name='states')
            self.op_actions = tf.placeholder(dtype=tf.float32, shape=[None, dim_action], name='actions')
            self.op_feed_params = tf.placeholder(dtype=tf.float32, shape=[None], name='feed_params')

            self.op_tangents = tf.placeholder(
                dtype=tf.float32, shape=[nn.utils.n_parameters(self.policy.parameters())])
            self.op_ent_coef = tf.placeholder(dtype=tf.float32, shape=[], name='ent_coef')

        self.op_mean_kl, self.op_loss, self.op_dist_std, self.op_dist_mean, self.op_policy_loss = \
            self(self.op_states, self.op_actions, self.op_advantages, self.op_ent_coef)

        self.op_sync_old, self.op_hessian_vec_prod, self.op_flat_grad = \
            self.compute_natural_grad(self.op_loss, self.op_mean_kl, self.op_tangents)

        self.op_vf_loss, self.op_train_vf = self.compute_vf(self.op_states, self.op_returns)

    def forward(self, states, actions, advantages, ent_coef):
        old_distribution: tf.distributions.Normal = self.old_policy(states)
        distribution: tf.distributions.Normal = self.policy(states)
        mean_kl = old_distribution.kl_divergence(distribution).reduce_sum(axis=1).reduce_mean()
        entropy = distribution.entropy().reduce_sum(axis=1).reduce_mean()
        entropy_bonus = ent_coef * entropy

        ratios: Tensor = (distribution.log_prob(actions) - old_distribution.log_prob(actions)) \
            .reduce_sum(axis=1).exp()
        # didn't output op_policy_loss since in principle it should be 0.
        policy_loss = ratios.mul(advantages).reduce_mean()

        # We're doing Gradient Ascent so this is, in fact, gain.
        loss = policy_loss + entropy_bonus

        return mean_kl, loss, distribution.stddev().log().reduce_mean().exp(), \
            distribution.mean().norm(axis=1).reduce_mean() / np.sqrt(10), policy_loss

    def compute_natural_grad(self, loss, mean_kl, tangents):
        params = self.policy.parameters()
        old_params = self.old_policy.parameters()
        hessian_vec_prod = nn.utils.hessian_vec_prod(mean_kl, params, tangents)
        flat_grad = nn.utils.parameters_to_vector(tf.gradients(loss, params))
        sync_old = tf.group(*[tf.assign(old_v, new_v) for old_v, new_v in zip(old_params, params)])

        return sync_old, hessian_vec_prod, flat_grad

    def compute_vf(self, states, returns):
        vf_loss = nn.MSELoss()(self.vf(states), returns).reduce_mean()
        optimizer = tf.train.AdamOptimizer(self.vf_lr)
        train_vf = optimizer.minimize(vf_loss)

        return vf_loss, train_vf

    @nn.make_method()
    def get_vf_loss(self, states, returns) -> List[np.ndarray]: pass

    @nn.make_method(fetch='sync_old')
    def sync_old(self) -> List[np.ndarray]: pass

    @nn.make_method(fetch='hessian_vec_prod')
    def get_hessian_vec_prod(self, states, tangents, actions) -> List[np.ndarray]: pass

    @nn.make_method(fetch='loss')
    def get_loss(self, states, actions, advantages, ent_coef) -> List[np.ndarray]: pass

    def train(self, ent_coef, samples, advantages, values):
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / np.maximum(advantages.std(), 1e-8)
        assert np.isfinite(advantages).all()
        self.sync_old()
        old_loss, grad, dist_std, mean_kl, dist_mean = self.get_loss(
            samples.state, samples.action, advantages, ent_coef, fetch='loss flat_grad dist_std mean_kl dist_mean')

        if np.allclose(grad, 0):
            logger.info('Zero gradient, not updating...')
            return

        def fisher_vec_prod(x):
            return self.get_hessian_vec_prod(samples.state, x, samples.action) + self.cg_damping * x

        assert np.isfinite(grad).all()
        nat_grad = conj_grad(fisher_vec_prod, grad, n_iters=self.n_cg_iters, verbose=False)

        assert np.isfinite(nat_grad).all()

        old_params = self.flatten.get_flat()
        step_size = np.sqrt(2 * self.max_kl / nat_grad.dot(fisher_vec_prod(nat_grad)))

        for _ in range(10):
            new_params = old_params + nat_grad * step_size
            self.flatten.set_flat(new_params)
            loss, mean_kl = self.get_loss(samples.state, samples.action, advantages, ent_coef, fetch='loss mean_kl')
            improve = loss - old_loss
            if not np.isfinite([loss, mean_kl]).all():
                logger.info('Got non-finite loss.')
            elif mean_kl > self.max_kl * 1.5:
                logger.info('Violated kl constraints, shrinking step... mean_kl = %.6f, max_kl = %.6f',
                            mean_kl, self.max_kl)
            elif improve < 0:
                logger.info("Surrogate didn't improve, shrinking step... %.6f => %.6f", old_loss, loss)
            else:
                break
            step_size *= 0.5
        else:
            logger.info("Couldn't find a good step.")
            self.flatten.set_flat(old_params)
        for param in self.policy.parameters():
            param.invalidate()

        # optimize value function
        vf_dataset = Dataset.fromarrays([samples.state, returns],
                                        dtype=[('state', ('f8', self.dim_state)), ('return_', 'f8')])
        vf_loss = self.train_vf(vf_dataset)

        return dist_mean, dist_std, vf_loss

    def train_vf(self, dataset: Dataset):
        for _ in range(self.n_vf_iters):
            for subset in dataset.iterator(64):
                self.get_vf_loss(subset.state, subset.return_, fetch='train_vf vf_loss')
        for param in self.parameters():
            param.invalidate()
        vf_loss = self.get_vf_loss(dataset.state, dataset.return_, fetch='vf_loss')
        return vf_loss


