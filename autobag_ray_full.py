import ray
import ray.tune as tune
from ray.rllib.agents import ppo
from envs.opamp_full_discrete import TwoStageAmp

ray.init()
tune.run_experiments({
    "opamp_full_discrete": {
        "checkpoint_freq":20,
        "run": "PPO",
        "env": TwoStageAmp,
        "stop": {"episode_reward_mean": -0.05},
        "config": {
	    #"observation_filter": "NoFilter",
            #"sample_batch_size":50,
            #"train_batch_size":1000,
            #"sgd_minibatch_size":42,
            #"num_sgd_iter":10,
            "lr":1e-3,
            "vf_loss_coeff":0.5,
            "horizon":  60, 
            "num_gpus": 1,
            "model":{"fcnet_hiddens": [128,128]},
            "num_workers": 5,
            "env_config":{"generalize":True},
            },
    },
})
