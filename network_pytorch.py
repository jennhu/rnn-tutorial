"""Definition of the network model and various RNN cells"""

from __future__ import division

import os
import numpy as np
import pickle

import torch
from torch import nn

from rnn_pytorch_huh import RNNLayer, RNNCell_continuous #, RNNCell_discrete

import tools_pytorch as tools



class Model(nn.Module):
    
    def __init__(self, model_dir, hp=None, sigma_rec=None, dt=None): # (self, nn_, N, nonlinear = None, Initializer = None, orth_gain = None, amp = None, powm = None, bias = False):
        """
        Initializing the model with information from hp: hyper-parameters

        Args:
            model_dir: string, directory of the model
            hp: a dictionary or None
            sigma_rec: if not None, overwrite the sigma_rec passed by hp
        """
        
        super().__init__()

        if hp is None:
            hp = tools.load_hp(model_dir)
        if sigma_rec is not None:
            hp['sigma_rec'] = sigma_rec
        if dt is not None:
            hp['dt'] = dt
            
        hp['decay'] = torch.exp( - torch.tensor(hp['dt']/hp['tau']))
        
        self.model_dir = model_dir
        self.hp = hp
        self.rng = np.random.RandomState(hp['seed'])
        torch.manual_seed(hp['seed'])
        np.random.seed(hp['seed'])      


# #         # Activation functions
# #         if hp['activation'] == 'power':             f_act = lambda x: tf.square(tf.nn.relu(x))
# #         elif hp['activation'] == 'retanh':          f_act = lambda x: tf.tanh(tf.nn.relu(x))
# #         elif hp['activation'] == 'relu+':           f_act = lambda x: tf.nn.relu(x + tf.constant(1.))
# #         else:                   f_act = getattr(tf.nn, hp['activation'])
  
# #         self.rnn = nn.RNNCell(hp['n_input'], hp['n_rnn'], nonlinearity='relu') # or 'tanh'
# #         self.rnn = nn.RNN(hp['n_input'], hp['n_rnn'], nonlinearity='relu', bias = False) # or 'tanh'
#         rnn0 = nn.RNN(hp['n_input'], hp['n_rnn'], nonlinearity='relu', bias = False) # or 'tanh'
#         self.rnn = rnn0
    
        self.rnn = RNNLayer(RNNCell_continuous, hp['n_input'], hp['n_rnn'], nn.ReLU()) # , hp['decay']
#         rnn_cell = RNNCell_continuous(hp['n_input'], hp['n_rnn'], decay = hp['decay'], nonlinearity=nn.ReLU(), bias = False) # or 'tanh'
#         rnn_cell = RNNCell_discrete(hp['n_input'], hp['n_rnn'], nonlinearity=nn.ReLU(), bias = False) # or 'tanh'
#         self.rnn = RNNLayer(rnn_cell)

#         for par0, par in zip(rnn0.parameters(),self.rnn.parameters()):
#             par.data = par0.data

        self.readout = nn.Linear(hp['n_rnn'], hp['n_output'], bias = False)
        self.loss_fnc = nn.MSELoss() if hp['loss_type'] == 'lsq' else nn.CrossEntropyLoss()

    def forward(self, trial):
        hp = self.hp
        x      = torch.tensor(trial.x)
        target = torch.tensor(trial.y)
        
        T, batch, _ = x.shape
        c_mask = torch.tensor(trial.c_mask).view(T, batch, -1)
        
        hidden0   = torch.zeros([1, batch, hp['n_rnn']])
        hidden, _ = self.rnn(x, hidden0)
        output    = self.readout(hidden)
        
        loss     = self.loss_fnc(c_mask * output, c_mask * target)
        loss_reg = (hidden.abs().mean() * hp['l1_h'] + hidden.norm() * hp['l2_h'])  #    Regularization terms
        
        for param in self.parameters():
            loss_reg += param.abs().mean() * hp['l1_weight'] + param.norm() * hp['l2_weight']

        return loss, loss_reg, output.detach().cpu().numpy()

        
#         # Create an optimizer.
#         hp['optimizer'] == 'adam' or 'sgd'
#         lr=hp['learning_rate'])
#         # Set cost
# #         self.set_optimizer()  # set cost
# MSELoss or Classification Error
# #         cost = self.loss + self.cost_reg + extra_cost

# #         # Variable saver
# #         # self.saver = tf.train.Saver(self.var_list)
# #         self.saver = tf.train.Saver()
        


#     def initialize(self):
#         """Initialize the model for training."""
#         sess = tf.get_default_session()
#         sess.run(tf.global_variables_initializer())

    def restore(self, load_dir=None):
        
        """restore the model"""
#         sess = tf.get_default_session()
        if load_dir is None:
            load_dir = self.model_dir
        save_path = os.path.join(load_dir, 'model.ckpt')
        try:
            self.saver.restore(sess, save_path)
        except:
            # Some earlier checkpoints only stored trainable variables
            self.saver = tf.train.Saver(self.var_list)
            self.saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

    def save(self):
#         import tensorflow as tf
#         self.saver = tf.train.Saver()
        """Save the model."""
        sess = tf.get_default_session()
        save_path = os.path.join(self.model_dir, 'model.ckpt')
        self.saver.save(sess, save_path)
        print("Model saved in file: %s" % save_path)

    def set_optimizer(self, extra_cost=None, var_list=None):
        raise NotImplementedError
        
#         cost = self.cost_lsq + self.cost_reg
#         if extra_cost is not None:
#             cost += extra_cost

#         if var_list is None:
#             var_list = self.var_list

#         self.grads_and_vars = self.opt.compute_gradients(cost, var_list)
#         # gradient clipping
#         capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var)
#                       for grad, var in self.grads_and_vars]
#         self.train_step = self.opt.apply_gradients(capped_gvs)

    def lesion_units(self, units, verbose=False):
        """Lesion units given by units

        Args:
            units : can be None, an integer index, or a list of integer indices
        """
        raise NotImplementedError
        
#         # Convert to numpy array
#         if units is None:
#             return
#         elif not hasattr(units, '__iter__'):
#             units = np.array([units])
#         else:
#             units = np.array(units)

#         # This lesioning will work for both RNN and GRU
#         n_input = self.hp['n_input']
#         for v in self.var_list:
#             if 'kernel' in v.name or 'weight' in v.name:
#                 # Connection weights
#                 v_val = sess.run(v)
#                 if 'output' in v.name:
#                     # output weights
#                     v_val[units, :] = 0
#                 elif 'rnn' in v.name:
#                     # recurrent weights
#                     v_val[n_input + units, :] = 0
#                 sess.run(v.assign(v_val))

#         if verbose:
#             print('Lesioned units:')
#             print(units)

########################


def is_weight(v):
    """Check if Tensorflow variable v is a connection weight."""
    return ('kernel' in v.name or 'weight' in v.name)

def popvec(y):
    """Population vector read out.

    Assuming the last dimension is the dimension to be collapsed

    Args:
        y: population output on a ring network. Numpy array (Batch, Units)

    Returns:
        Readout locations: Numpy array (Batch,)
    """
    pref = np.arange(0, 2*np.pi, 2*np.pi/y.shape[-1])  # preferences
    temp_sum = y.sum(axis=-1)
    temp_cos = np.sum(y*np.cos(pref), axis=-1)/temp_sum
    temp_sin = np.sum(y*np.sin(pref), axis=-1)/temp_sum
    loc = np.arctan2(temp_sin, temp_cos)
    return np.mod(loc, 2*np.pi)


def get_perf(y_hat, y_loc):
    """Get performance.

    Args:
      y_hat: Actual output. Numpy array (Time, Batch, Unit)
      y_loc: Target output location (-1 for fixation).
        Numpy array (Time, Batch)

    Returns:
      perf: Numpy array (Batch,)
    """
    if len(y_hat.shape) != 3:
        raise ValueError('y_hat must have shape (Time, Batch, Unit)')
    # Only look at last time points
    y_loc = y_loc[-1]
    y_hat = y_hat[-1]

    # Fixation and location of y_hat
    y_hat_fix = y_hat[..., 0]
    y_hat_loc = popvec(y_hat[..., 1:])

    # Fixating? Correctly saccading?
    fixating = y_hat_fix > 0.5

    original_dist = y_loc - y_hat_loc
    dist = np.minimum(abs(original_dist), 2*np.pi-abs(original_dist))
    corr_loc = dist < 0.2*np.pi

    # Should fixate?
    should_fix = y_loc < 0

    # performance
    perf = should_fix * fixating + (1-should_fix) * corr_loc * (1-fixating)
    return perf



