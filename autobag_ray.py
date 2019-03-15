import ray
import ray.tune as tune
from ray.rllib.agents import ppo
from ray.rllib.contrib.random_agent.random_agent import RandomAgent
from envs.discrete_opamp import TwoStageAmp

ray.init()
tune.run_experiments({
    "my_experiment": {
        "checkpoint_freq":2,
        #"restore": "/home/ksettaluri6/ray_results/my_experiment/PPO_TwoStageAmp_0_2019-03-08_02-27-540yzzmiqv/checkpoint_260/checkpoint-260",
        "run": "PPO",
        "env": TwoStageAmp,
        "stop": {"episode_reward_mean": -0.02},
        "config": {
	    #"observation_filter": "NoFilter",
            "sample_batch_size": 200,
            "train_batch_size": 1200,
            "sgd_minibatch_size": 1200,
            "num_sgd_iter":3,
            "lr":1e-3,
            "vf_loss_coeff":0.5,
            "horizon":  60,#tune.grid_search([15,25]),
            "num_gpus": 1,
            "model":{"fcnet_hiddens": [64, 64]},
            "num_workers": 6,
            "env_config":{"generalize":True, "save_specs":True},
            },
    },
})
