import ray
import ray.tune as tune
from ray.rllib.agents import ppo
from envs.discrete_opamp import TwoStageAmp

ray.init()
tune.run_experiments({
    "my_experiment": {
        "checkpoint_freq":1,
        "run": "PPO",
        "env": TwoStageAmp,
        "stop": {"episode_reward_mean": -0.1},
        "config": {
	    #"observation_filter": "NoFilter",
            "sample_batch_size": 30,
            "train_batch_size": 210,
            "sgd_minibatch_size": 70,
            "horizon": 50,
            "num_gpus": 0,
            "num_workers": 0,
            },
    },
})
