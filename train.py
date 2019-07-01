""" Main training loop.
Copied from https://github.com/gyyang/multitask. Modified to work with pytorch instead of tensorflow framework. 
"""

from __future__ import division

import sys
import time
from collections import defaultdict

import torch
import math
import numpy as np

from utils import get_perf
import task
# import tools
# from analysis import variance



print_flag = False





def gen_feed_dict(trial, hp):
    n_time, batch_size = trial.x.shape[:2]
    if hp['in_type'] == 'normal':
        pass
    elif hp['in_type'] == 'multi':
        new_shape = [n_time,    batch_size,     hp['rule_start']*hp['n_rule']]

        x = np.zeros(new_shape, dtype=np.float32)
        for i in range(batch_size):
            ind_rule = np.argmax(trial.x[0, i, hp['rule_start']:])
            i_start = ind_rule*hp['rule_start']
            x[:, i, i_start:i_start+hp['rule_start']] = trial.x[:, i, :hp['rule_start']]
        trial.x     = x
    else:
        raise ValueError()
        
    trial.x      = torch.tensor(trial.x)
    trial.y      = torch.tensor(trial.y)
    trial.c_mask = torch.tensor(trial.c_mask).view(n_time,  batch_size, -1)

    return trial



def get_default_hp(ruleset):
    '''Get a default hp.

    Useful for debugging.

    Returns:
        hp : a dictionary containing training hpuration
    '''
    num_ring = task.get_num_ring(ruleset)
    n_rule   = task.get_num_rule(ruleset)

    n_eachring = 32
    n_input, n_output = 1+num_ring*n_eachring+n_rule, n_eachring+1
    hp = {
            # batch size for training
            'batch_size_train': 64,
            # batch_size for testing
            'batch_size_test': 512,
            # input type: normal, multi
            'in_type': 'normal',
            # Type of RNNs: LeakyRNN, LeakyGRU, EILeakyGRU, GRU, LSTM
            'rnn_type': 'LeakyRNN',
            # whether rule and stimulus inputs are represented separately
            'use_separate_input': False,
            # Type of loss functions
            'loss_type': 'lsq',
            # Optimizer
            'optimizer': 'adam',
            # Type of activation runctions, relu, softplus, tanh, elu
            'activation': 'relu',
            # Time constant (ms)
            'tau': 100,
            # discretization time step (ms)
            'dt': 20,
            # discretization time step/time constant
            'alpha': 0.2,
            # recurrent noise
            'sigma_rec': 0.05,
            # input noise
            'sigma_x': 0.01,
            # leaky_rec weight initialization, diag, randortho, randgauss
            'w_rec_init': 'randortho',
            # a default weak regularization prevents instability
            'l1_h': 0,
            # l2 regularization on activity
            'l2_h': 0,
            # l2 regularization on weight
            'l1_weight': 0,
            # l2 regularization on weight
            'l2_weight': 0,
            # l2 regularization on deviation from initialization
            'l2_weight_init': 0,
            # proportion of weights to train, None or float between (0, 1)
            'p_weight_train': None,
            # Stopping performance
            'target_perf': 1.,
            # number of units each ring
            'n_eachring': n_eachring,
            # number of rings
            'num_ring': num_ring,
            # number of rules
            'n_rule': n_rule,
            # first input index for rule units
            'rule_start': 1+num_ring*n_eachring,
            # number of input units
            'n_input': n_input,
            # number of output units
            'n_output': n_output,
            # number of recurrent units
            'n_rnn': 256,
            # number of input units
            'ruleset': ruleset,
            # name to save
            'save_name': 'test',
            # learning rate
            'learning_rate': 0.001,
            # intelligent synapses parameters, tuple (c, ksi)
            'c_intsyn': 0,
            'ksi_intsyn': 0,
            }

    return hp



def display_rich_output(model, step, log, model_dir):
    """Display step by step outputs during training."""
    variance._compute_variance_bymodel(model)
    rule_pair = ['contextdm1', 'contextdm2']
    save_name = '_atstep' + str(step)
    title = ('Step ' + str(step) +  ' Perf. {:0.2f}'.format(log['perf_avg'][-1]))
    variance.plot_hist_varprop(model_dir, rule_pair,  figname_extra=save_name,  title=title)
    plt.close('all')


def set_hyperparameters(model_dir,
          hp=None,
          max_steps=1e7,
          display_step=500,
          ruleset='mante',
          rule_trains=None,
          rule_prob_map=None,
          seed=0,
          rich_output=False,
          load_dir=None,
          trainables=None,
          ):
    """Train the network.

    Args:
        model_dir: str, training directory
        hp: dictionary of hyperparameters
        max_steps: int, maximum number of training steps
        display_step: int, display steps
        ruleset: the set of rules to train
        rule_trains: list of rules to train, if None then all rules possible
        rule_prob_map: None or dictionary of relative rule probability
        seed: int, random seed to be used

    Returns:
        model is stored at model_dir/model.ckpt    : Not implemented
        training configuration is stored at model_dir/hp.json
    """

#     tools.mkdir_p(model_dir)

    # Network parameters
    default_hp = get_default_hp(ruleset)
    if hp is not None:
        default_hp.update(hp)
    hp = default_hp
    hp['seed'] = seed
    hp['rng'] = np.random.RandomState(seed)
    hp['model_dir']    = model_dir
    hp['max_steps']    = max_steps
    hp['display_step'] = display_step
    hp['rich_output']  = rich_output
    hp['decay']        = math.exp( -  hp['dt']/hp['tau'])  # 1 - hp['dt']/hp['tau'] 

    # Rules to train and test. Rules in a set are trained together
    if rule_trains is None:
        # By default, training all rules available to this ruleset
        hp['rule_trains'] = task.rules_dict[ruleset]
    else:
        hp['rule_trains'] = rule_trains
    hp['rules'] = hp['rule_trains']
    

    # Assign probabilities for rule_trains.
    if rule_prob_map is None:
        rule_prob_map = dict()

    # Turn into rule_trains format
    hp['rule_probs'] = None
    if hasattr(hp['rule_trains'], '__iter__'):
        # Set default as 1.
        rule_prob = np.array(
                [rule_prob_map.get(r, 1.) for r in hp['rule_trains']])
        hp['rule_probs'] = list(rule_prob/np.sum(rule_prob))
        
#     tools.save_hp(hp, model_dir)     # saving model: not implemented

#     ##### Build the model #####
#     model = Model(model_dir, hp=hp)  # model is defined outside. 
    
    if print_flag:                     # Display hp 
        for key, val in hp.items():
            print('{:20s} = '.format(key) + str(val))

    if load_dir is not None:
        raise NotImplementedError            # loading saved model: not implemented

    # penalty on deviation from initial weight
    if hp['l2_weight_init'] > 0:
        raise NotImplementedError

    # partial weight training
    if ('p_weight_train' in hp    and   (hp['p_weight_train'] is not None)    and      hp['p_weight_train'] < 1.0):
        raise NotImplementedError
        
    
    if  hp['optimizer'] == 'adam':
        optimizer = torch.optim.Adam
    elif hp['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD
    else: 
        raise NotImplementedError
    # Store results
    log = defaultdict(list)
    log['model_dir'] = model_dir
    
    return hp, log, optimizer #, model



def train(run_model, optimizer, hp, log):

    optim = optimizer(run_model.model.parameters(), lr=hp['learning_rate'])

    step = 0
    t_start = time.time()
    while step * hp['batch_size_train'] <= hp['max_steps']:
        try:
            # Validation
            if step % hp['display_step'] == 0:
                log['trials'].append(step * hp['batch_size_train'])
                log['times'].append(time.time()-t_start)
                log = do_eval(run_model, log, hp['rule_trains'])
                #if log['perf_avg'][-1] > hp['target_perf']:
                #check if minimum performance is above target    
                if log['perf_min'][-1] > hp['target_perf']:
                    print('Perf reached the target: {:0.2f}'.format( hp['target_perf']))
                    break

                if hp['rich_output']:
                    display_rich_output(run_model, step, log, hp['model_dir'])

            # Training
            rule_train_now = hp['rng'].choice(hp['rule_trains'],   p=hp['rule_probs'])
            
            optim.zero_grad()
            c_lsq, c_reg, _, _, _ = run_model(rule = rule_train_now, batch_size = hp['batch_size_train'])
            loss = c_lsq + c_reg
            loss.backward()
            optim.step()
            step += 1

        except KeyboardInterrupt:
            print("Optimization interrupted by user")
            break

    print("Optimization finished!")



def do_eval(run_model, log, rule_train):
    """Do evaluation.

    Args:
        model: Model class instance
        log: dictionary that stores the log
        rule_train: string or list of strings, the rules being trained
    """
    hp = run_model.hp
    if not hasattr(rule_train, '__iter__'):
        rule_name_print = rule_train
    else:
        rule_name_print = ' & '.join(rule_train)

    print('Trial {:7d}'.format(log['trials'][-1]) + '  | Time {:0.2f} s'.format(log['times'][-1]) +  '  | Now training '+rule_name_print)

    for rule_test in hp['rules']:
        n_rep = 16
        batch_size_test_rep = int(hp['batch_size_test']/n_rep)
        clsq_tmp, creg_tmp, perf_tmp = list(), list(), list()
        
        for i_rep in range(n_rep):
            with torch.no_grad():
                c_lsq, c_reg, y_hat_test, _, trial = run_model(rule = rule_test, batch_size = batch_size_test_rep)

            # Cost is first summed over time, and averaged across batch and units. We did the averaging over time through c_mask
            perf_test = np.mean(get_perf(y_hat_test, trial.y_loc))
            clsq_tmp.append(c_lsq)
            creg_tmp.append(c_reg)
            perf_tmp.append(perf_test)

        log['cost_'+rule_test].append(np.mean(clsq_tmp, dtype=np.float64))
        log['creg_'+rule_test].append(np.mean(creg_tmp, dtype=np.float64))
        log['perf_'+rule_test].append(np.mean(perf_tmp, dtype=np.float64))
        print('{:15s}'.format(rule_test) + '| cost {:0.6f}'.format(np.mean(clsq_tmp)) + '| c_reg {:0.6f}'.format(np.mean(creg_tmp)) +  '  | perf {:0.2f}'.format(np.mean(perf_tmp)))
        sys.stdout.flush()

    # TODO: This needs to be fixed since now rules are strings
    if hasattr(rule_train, '__iter__'):
        rule_tmp = rule_train
    else:
        rule_tmp = [rule_train]
    perf_tests_mean = np.mean([log['perf_'+r][-1] for r in rule_tmp])
    log['perf_avg'].append(perf_tests_mean)

    perf_tests_min = np.min([log['perf_'+r][-1] for r in rule_tmp])
    log['perf_min'].append(perf_tests_min)

    # Saving the model
#     run_model.save()           # Saving not implemented
#     tools.save_log(log)

    return log



#######################

# if __name__ == '__main__':
#     import argparse
#     import os
#     parser = argparse.ArgumentParser(
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#     parser.add_argument('--modeldir', type=str, default='data/debug')
#     args = parser.parse_args()

#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     hp = {'activation': 'softplus',
#           'n_rnn': 64,
#           'mix_rule': True,
#           'l1_h': 0.,
#           'use_separate_input': True}
#     train(args.modeldir,
#           seed=1,
#           hp=hp,
#           ruleset='all',
#           rule_trains=['contextdelaydm1', 'contextdelaydm2',
#                                  'contextdm1', 'contextdm2'],
#           display_step=500)