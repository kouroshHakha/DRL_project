from train_pg import *
import gym
import gym_ckt
import IPython
import time

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='rnn_vpg')
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--norm_adv', '-na', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--hist_dim', '-hd', type=int, default=128)
    parser.add_argument('--state_dim', '-sd', type=int, default=128)
    parser.add_argument('--mini_batch', '-mb', type=int, default=16)
    parser.add_argument('--rollout', '-ro', type=int, default=20)
    parser.add_argument('--lr', '-lr', type=float, default=0.001)
    parser.add_argument('--gamma', '-g', type=float, default=0.99)
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    env = gym.make('ckt-v0')

    n_iter = 100

    computation_graph_args = {
        'ob_dim': env.observation_space.shape[0],
        'ac_dim': env.action_space.shape[0],
        'hist_dim': args.hist_dim,
        'state_dim': args.state_dim,
        'mini_batch_size': args.mini_batch,
        'roll_out_h': args.rollout,
        'learning_rate': args.lr,
    }

    pg_flavor_args = {
        'gamma': args.gamma,
        'reward_to_go': args.reward_to_go,
        'nn_baseline': args.nn_baseline,
        'normalize_advantages': args.norm_adv,
    }
    agent = Agent(
        env=env,
        computation_graph_args=computation_graph_args,
        pg_flavor_args=pg_flavor_args,

    )

    agent.build_computation_graph()

    # tensorflow: config, session, variable initialization
    agent.init_tf_sess()

    agent.train(n_iter, os.path.join(logdir, '%d'%0))
