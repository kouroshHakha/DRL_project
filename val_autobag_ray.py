import ray
import ray.tune as tune
from ray.rllib.agents import ppo
from ray.rllib.contrib.random_agent.random_agent import RandomAgent
#from envs.discrete_opamp import TwoStageAmp
#from envs.bag_opamp_discrete import TwoStageAmp
from envs.bag_tia_discrete import TIA

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', '-cpd', type=str)
args = parser.parse_args()

ray.init()

config_validation = {
            "sample_batch_size": 200,
            "train_batch_size": 1200,
            "sgd_minibatch_size": 1200,
            "num_sgd_iter":3,
            "lr":1e-3,
            "vf_loss_coeff":0.5,
            "horizon":  60,#tune.grid_search([15,25]),
            "num_gpus": 0,
            "model":{"fcnet_hiddens": [64, 64]},
            "num_workers": 1,
            "env_config":{"generalize":False, "save_specs":False, "run_valid":True},
            }

config_train = {
            "sample_batch_size": 20,
            "train_batch_size": 120,
            "sgd_minibatch_size": 120,
            "num_sgd_iter": 3,
            "lr":1e-3,
            "vf_loss_coeff": 0.5,
            "horizon":  20,
            "num_gpus": 0,
            "model":{"fcnet_hiddens": [64, 64]},
            "num_workers": 6,
            "env_config":{"generalize":True, "save_specs":True},
            }

if not args.checkpoint_dir:
    trials = tune.run_experiments({
        "train_ppo": {
        "checkpoint_freq":1,
        "run": "PPO",
        "env": TIA,
#        "stop": {"training_iteration": 3, "episode_reward_max": -0.02},
        "stop": {"episode_reward_mean": 0.0},
        "config": config_train},
    })
else:
    print("RESTORING NOW!!!!!!")
    #print(trials[0]._checkpoint.value)
    tune.run_experiments({
        "restore_ppo": {
        "run": "PPO",
        "config": config_validation,
        "env": TwoStageAmp,
        #"restore": trials[0]._checkpoint.value},
        "restore": args.checkpoint_dir,
        "checkpoint_freq":2},
    })
