from multiprocessing import Process
from rl_rinokeras import train

from envs.pointmass4 import PointMass as PointMass_v4
from envs.pointmass5_seq_with_int_rewards import PointMass as PointMass_v5
from envs.ckt_env_discrete import CSAmp as CSAmpDiscrete
from envs.goal_env_wrapper import GoalEnvWrapper
from envs.pointmass3d_discrete import PointMass3dd
from envs.pointmass4_cont import PointMass4Cont
import gym
from gym.envs.registration import register

def main():
    import argparse
    parser = argparse.ArgumentParser('Main Example')
    parser.add_argument('env', type=str, help='Which gym environment to run on')
    parser.add_argument('exp_name', type=str, help='The name of the experiment')
    parser.add_argument('--policy', '-p' ,type=str, choices=['standard', 'lstm'], default='standard',
                        help='Which type of policy to run')
    parser.add_argument('--alg', '-a', type=str, choices=['vpg', 'ppo'], default='vpg',
                        help='Which algorithm to use to train the agent')
    parser.add_argument('--logstd', type=float, default=0, help='initial_logstd')
    parser.add_argument('--seed', '-s', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1, help='number of experiments')
    parser.add_argument('--sparse', action='store_true', help='determines sparsity of the reward')
    parser.add_argument('--her', action='store_true', help='Added Hindsight Experience Buffer')
    parser.add_argument('--max_iter', '-n', type=int, default=500)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--max_ep_len', '-ep', type=float, default=50)
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--model_dim', '-size', type=int, default=64)
    parser.add_argument('--g_star_batch_size', '-g_star_b', type=int, default=40)
    parser.add_argument('--sub_g_batch_size', '-sub_g_b', type=int, default=40)
    parser.add_argument('--ent_coeff', '-ec', type=float, default=1)
    parser.add_argument('--buffer_size', '-bs', type=int, default=40)
    parser.add_argument('--sub_goal_strategy', type=str, choices=['last', 'random', 'best'], default='random')
    parser.add_argument('--scale', type=int, default=200)
    parser.add_argument('--l2_scale', '-l2', type=float, default=0.0)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--n_updates_per_sub_goals', '-nups', type=int, default=5)
    args = parser.parse_args()


    env_sub_goals = None
    env_actual_goal = None
    monitor_period = None

    if args.env == 'pm4':
        env_sub_goals = PointMass_v4(sparse=args.sparse, scale=args.scale)
        env_actual_goal = PointMass_v4(sparse=args.sparse, scale=args.scale)
    elif args.env == 'ckt-v2':
        env_sub_goals = CSAmpDiscrete(sparse=args.sparse)
        env_actual_goal = CSAmpDiscrete(sparse=args.sparse)
    elif args.env == 'pm3dd':
        env_sub_goals = PointMass3dd(sparse=args.sparse, scale=args.scale)
        env_actual_goal = PointMass3dd(sparse=args.sparse, scale=args.scale)
    elif args.env == 'pm4c':
        env_sub_goals = PointMass4Cont(sparse=args.sparse, scale=args.scale)
        env_actual_goal = PointMass4Cont(sparse=args.sparse, scale=args.scale)
    else:
        env_sub_goals = gym.make(args.env, reward_type='sparse' if args.sparse else 'dense')
        env_actual_goal = gym.make(args.env, reward_type='sparse' if args.sparse else 'dense')
        env_sub_goals = GoalEnvWrapper(env_sub_goals)
        env_actual_goal = GoalEnvWrapper(env_actual_goal)
        monitor_period = 100

    processes = []
    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)

        def train_func():
            train(
                exp_name=args.exp_name,
                env_name=args.env,
                env_actual_goal=env_actual_goal,
                env_sub_goals=env_sub_goals,
                seed=seed,
                policy_type=args.policy,
                algorithm=args.alg,
                init_logstd=args.logstd,
                use_her=args.her,
                gamma=args.gamma,
                model_dim=args.model_dim,
                n_layers=args.n_layers,
                n_rollout_per_actual_goal=args.g_star_batch_size,
                n_rollouts_per_sub_goals=args.sub_g_batch_size,
                max_ep_steps=args.max_ep_len,
                entcoeff = args.ent_coeff,
                ex_buffer_size=args.buffer_size,
                max_iter=args.max_iter,
                sub_goal_strategy=args.sub_goal_strategy,
                l2_scale=args.l2_scale,
                learning_rate=args.learning_rate,
                n_updates_per_sub_goals=args.n_updates_per_sub_goals,
                monitor_period=monitor_period
            )
        train_func()
        # # Awkward hacky process runs, because Tensorflow does not like
        # # repeatedly calling train_PG in the same thread.
        # p = Process(target=train_func, args=tuple())
        # p.start()
        # processes.append(p)
        # if you comment in the line below, then the loop will block
        # until this process finishes
        # p.join()

    # for p in processes:
    #     p.join()

if __name__ == "__main__":
    main()