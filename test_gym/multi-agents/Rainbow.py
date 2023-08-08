import time
import gym

from models.CartPole.Rainbow import CartPole

"""
Created on Tue Apr 27 2:19:47 2023
@author: hien
"""
from itertools import count
import logging
import os
import torch
from tqdm import tqdm
from src.utils import *
log = logging.getLogger(__name__)
from argparse import ArgumentParser

from algorithms.Rainbow import Rainbow
import gym

def argument_parser():
    parser = ArgumentParser()
    # Game options
    parser.add_argument('--show-screen', type=bool, default=True)
    parser.add_argument('--render-last', type=bool, default=True)
    parser.add_argument('--figure-path', type=str, default='figures/')
    
    # DDQN arguments
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--beta', type=float, default=0.6)
    parser.add_argument('--prior_eps', type=float, default=1e-6)
    parser.add_argument('--n-step', type=int, default=4)
    
    # model training arguments
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--memory-size', type=int, default=8192)
    parser.add_argument('--num-episodes', type=int, default=2000)
    parser.add_argument('--model-path', type=str, default='tmp/model.pt')
    parser.add_argument('--load-model', action='store_true')
    
    return parser.parse_args()

def main():
    args = argument_parser()
    if args.show_screen:
        mode = 'human'
    else:
        mode = 'rgb_array'
        
    env = gym.make('ma_gym:Combat-v0')
    n_actions = env.action_space[0].n
    n_observations = env.observation_space[0].shape[0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    algorithm = None
    set_seed(1)
    
    model = CartPole(
        n_observations=n_observations,
        n_actions=n_actions,
        atom_size=41, 
        v_min=-15,
        v_max=30,
        optimizer=args.optimizer,
        lr=args.lr,
        device=device
    )
    
    model = model.to(device)
    
    algorithm = Rainbow(   
        n_observations=n_observations, 
        n_actions=n_actions,
        model=model,
        tau=args.tau,
        gamma=args.gamma,
        memory_size=args.memory_size,
        model_path=args.model_path,
        batch_size=args.batch_size,
        alpha=args.alpha,
        beta=args.beta,
        prior_eps=args.prior_eps,
        n_step=args.n_step,
        v_min=-15,
        v_max=30,
        atom_size=41,
    )
    
    algorithm.set_multi_agent_env(env.n_agents)
    
    if args.model_path:
        model_dir = os.path.dirname(args.model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            logging.info('Created model directory: {}'.format(model_dir))
        if args.load_model:
            algorithm.load_model(args.model_path)
    
    args.figure_path = os.path.join(args.figure_path, 'Rainbow')
    
    if not os.path.exists(args.figure_path):
        os.makedirs(args.figure_path)
        
    print("History and ep_rewards saved at {}".format(args.figure_path))
        
    ep_rewards = []
    
    for episode in tqdm(range(args.num_episodes)):
        obs_n = env.reset()
        env.render()
        # cnt = 0
        done_n = [False for _ in range(env.n_agents)]
        ep_reward = 0
        while not all(done_n):
            actions = [algorithm.get_action(obs_i) for obs_i in obs_n]
            opp_health_preaction = np.array(list(env.opp_health.values()))
            next_obs_n, reward_n, done_n, info = env.step(actions)
            curr_opp_health = np.array(list(env.opp_health.values()))
            opp_health_reduce_mean = np.mean(opp_health_preaction - curr_opp_health)
            for i in range(env.n_agents):
                reward_n[i] = reward_n[i] + opp_health_reduce_mean
                if info['health'][i] > 0:
                    reward_n[i] += 0.01 * info['health'][i]
            env.render()
            ep_reward += sum(reward_n)
            for state, action, reward, next_state, done in zip(obs_n, actions, reward_n, next_obs_n, done_n):
                transition = [state, action, reward, next_state, done]
                one_step_transition = algorithm.memory_n.store(*transition)
                if one_step_transition:
                    algorithm.memorize(*one_step_transition)
                algorithm.memorize(*transition)
            obs_n = next_obs_n
        
        ep_rewards.append(ep_reward)
        
        if episode % 3 == 0 and algorithm.fully_mem(0.25):
            history_loss = algorithm.replay(args.batch_size)
            plot_timeseries(history_loss, args.figure_path, 'episode', 'loss', 'Training Loss')
            plot_timeseries(ep_rewards, args.figure_path, 'episode', 'ep_rewards', 'Training Timesteps')
            if ep_rewards[-1] >= max(ep_rewards[:-1]):
                algorithm.save_model()
                
                
                
if __name__ == "__main__":
    main()