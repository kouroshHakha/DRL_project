import ray
import ray.tune as tune
from ray.rllib.agents import ppo
from ray.rllib.contrib.random_agent.random_agent import RandomAgent
from envs.discrete_opamp import TwoStageAmp

ray.init()
tune.run_experiments({
    "my_experiment": {
        "checkpoint_freq":2,
        "restore": "/home/ksettaluri6/ray_results/my_experiment/PPO_TwoStageAmp_0_2019-03-13_11-13-51i3_j_z4h/checkpoint_438/checkpoint-438",
        "run": "PPO",
        "env": TwoStageAmp,
        "stop": {"episode_reward_max": -0.02},
        "config": {
	    #"observation_filter": "NoFilter",
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
            "env_config":{"generalize":False, "save_specs":False},
            },
    },
})
