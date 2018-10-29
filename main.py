from train_pg import *
import gym
import gym_ckt
import IPython
if __name__ == '__main__':
    env = gym.make('ckt-v0')

    n_iter = 100

    computation_graph_args = {
        'ob_dim': env.observation_space.shape[0],
        'ac_dim': env.action_space.shape[0],
        'hist_dim': 128,
        'state_dim': 128,
        'mini_batch_size': 16,
        'roll_out_h': 20,
        'learning_rate': 0.001,
    }

    pg_flavor_args = {
        'gamma': 0.99,
        'reward_to_go': False,
        'nn_baseline': False,
        'normalize_advantages': False,
    }
    agent = Agent(
        env=env,
        computation_graph_args=computation_graph_args,
        pg_flavor_args=pg_flavor_args,

    )

    agent.build_computation_graph()

    # tensorflow: config, session, variable initialization
    agent.init_tf_sess()

    agent.train(n_iter)
