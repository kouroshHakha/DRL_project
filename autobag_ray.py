import ray
import ray.tune as tune
from ray.rllib.agents import ppo
from envs.opamp_discrete import TwoStageAmp

ray.init()
tune.run_experiments({
    "my_experiment": {
        "run": "PPO",
        "env": TwoStageAmp,
        "stop": {"episode_reward_mean": 50},
        "config": {
            "sample_batch_size": 30,
            "train_batch_size": 210,
            "sgd_minibatch_size": 70,
            "horizon": 50,
            "num_gpus": 0,
            "num_workers": 3,
        },
    },
})
