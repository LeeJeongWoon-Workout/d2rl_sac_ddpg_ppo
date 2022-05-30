from configparser import ConfigParser
from argparse import ArgumentParser

import torch
import gym
import numpy as np
import os
import matplotlib.pyplot as plt
from agents.d2rl_ppo import D2RL_PPO
from agents.d2rl_ddpg import D2RL_DDPG
from agents.d2rl_sac import D2RL_SAC
from agents.ppo import PPO
from agents.sac import SAC
from agents.ddpg import DDPG
import pandas as pd
import seaborn as sns
import itertools
from utils.utils import make_transition, Dict, RunningMeanStd
os.makedirs('./model_weights', exist_ok=True)

parser = ArgumentParser('parameters')

parser.add_argument("--env_name", type=str, default = 'Hopper-v2', help = "'Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','HumanoidStandup-v2',\
          'InvertedDoublePendulum-v2', 'InvertedPendulum-v2' (default : Hopper-v2)")
parser.add_argument("--algo", type=str, default = 'ppo', help = 'algorithm to adjust (default : ppo)')
parser.add_argument('--train', type=bool, default=True, help="(default: True)")
parser.add_argument('--render', type=bool, default=False, help="(default: False)")
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs, (default: 1000)')
parser.add_argument('--tensorboard', type=bool, default=False, help='use_tensorboard, (default: False)')
parser.add_argument("--load", type=str, default = 'no', help = 'load network name in ./model_weights')
parser.add_argument("--save_interval", type=int, default = 100, help = 'save interval(default: 100)')
parser.add_argument("--print_interval", type=int, default = 1, help = 'print interval(default : 20)')
parser.add_argument("--use_cuda", type=bool, default = True, help = 'cuda usage(default : True)')
parser.add_argument("--reward_scaling", type=float, default = 0.1, help = 'reward scaling(default : 0.1)')
parser.add_argument('--iteration',type=int,default=3,metavar='G')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
args = parser.parse_args()
parser = ConfigParser()
parser.read('/home/airlab/PycharmProjects/pythonProject5/Mujoco-Pytorch/config.ini')
agent_args = Dict(parser,args.algo)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.use_cuda == False:
    device = 'cpu'

if args.tensorboard:
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter()
else:
    writer = None

env = gym.make(args.env_name)
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
state_rms = RunningMeanStd(state_dim)

test_env=list()
for i in range(args.iteration):
    test=gym.make(args.env_name)
    test.seed(1234+i)
    test_env.append(test)

if args.algo == 'ppo':
    agent1 = D2RL_PPO(writer, device, state_dim, action_dim, agent_args,layer_num=5)
    agent2 = PPO(writer, device, state_dim, action_dim, agent_args, layer_num=4)
    agent3 = PPO(writer, device, state_dim, action_dim, agent_args, layer_num=5)
    agent4 = PPO(writer, device, state_dim, action_dim, agent_args, layer_num=6)
    agent5 = PPO(writer, device, state_dim, action_dim, agent_args, layer_num=8)


elif args.algo == 'sac':
    agent1 = D2RL_SAC(writer, device, state_dim, action_dim, agent_args,layer_num=5)
    agent2 = SAC(writer, device, state_dim, action_dim, agent_args, layer_num=4)
    agent3 = SAC(writer, device, state_dim, action_dim, agent_args, layer_num=5)
    agent4 = SAC(writer, device, state_dim, action_dim, agent_args, layer_num=6)
    agent5 = SAC(writer, device, state_dim, action_dim, agent_args, layer_num=8)

elif args.algo == 'ddpg':
    from utils.noise import OUNoise

    noise1 = OUNoise(action_dim, 0)
    noise2 = OUNoise(action_dim, 0)
    noise3 = OUNoise(action_dim, 0)
    noise4 = OUNoise(action_dim, 0)
    noise5 = OUNoise(action_dim, 0)

    agent1 = D2RL_DDPG(writer, device, state_dim, action_dim, agent_args,noise1,layer_num=5)
    agent2 = DDPG(writer, device, state_dim, action_dim, agent_args,noise2, layer_num=4)
    agent3 = DDPG(writer, device, state_dim, action_dim, agent_args,noise3, layer_num=5)
    agent4 = DDPG(writer, device, state_dim, action_dim, agent_args, noise4,layer_num=6)
    agent5 = DDPG(writer, device, state_dim, action_dim, agent_args, noise5,layer_num=8)

if (torch.cuda.is_available()) and (args.use_cuda):
    agent1 = agent1.cuda()
    agent2 = agent1.cuda()
    agent3 = agent1.cuda()
    agent4 = agent1.cuda()
    agent5 = agent1.cuda()

agent_lst=[agent1,agent2,agent3,agent4,agent5]
num_lst=list()
reward_lst=list()

for i in range(5):
    n=list()
    r=list()
    num_lst.append(n)
    reward_lst.append(r)



for i in range(5):

    agent=agent_lst[i]
    total_num=0

    if agent_args.on_policy == True:
        state_ = (env.reset())
        state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
        epi=0
        while True:
            if total_num > args.num_steps:
                break
            for t in range(agent_args.traj_length):
                if total_num > args.num_steps:
                    break
                if args.render:
                    env.render()
                mu, sigma = agent.get_action(torch.from_numpy(state).float().to(device))
                dist = torch.distributions.Normal(mu, sigma[0])
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1, keepdim=True)
                next_state_, reward, done, info = env.step(action.cpu().numpy())
                next_state = np.clip((next_state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
                transition = make_transition(state, \
                                             action.cpu().numpy(), \
                                             np.array([reward * args.reward_scaling]), \
                                             next_state, \
                                             np.array([done]), \
                                             log_prob.detach().cpu().numpy() \
                                             )
                agent.put_data(transition)
                total_num += 1
                if done:
                    epi+=1
                    state_ = (env.reset())
                    state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)

                else:
                    state = next_state
                    state_ = next_state_

                if epi % 5 == 0:
                    for iter in range(args.iteration):

                        episode_reward = 0
                        episode_steps = 0
                        done = False
                        state = test_env[iter].reset()
                        state = np.clip((state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
                        print("total_num_step: {} , algorithm: {}  Evaluating: {}".format(total_num, i, iter))
                        while not done:
                            mu, sigma = agent.get_action(torch.from_numpy(state).float().to(device))
                            dist = torch.distributions.Normal(mu, sigma[0])
                            action = dist.sample()
                            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
                            next_state_, reward, done, info = env.step(action.cpu().numpy())
                            next_state = np.clip((next_state_ - state_rms.mean) / (state_rms.var ** 0.5 + 1e-8), -5, 5)
                            state = next_state_

                            episode_steps += 1
                            episode_reward += reward

                        reward_lst[i].append(episode_reward)
                        num_lst[i].append(total_num)

            agent.train_net()












    else:  # off policy
        for iteration in itertools.count(1):
            if total_num > args.num_steps:
                break
            state = env.reset()
            done = False
            while not done:
                total_num += 1
                if args.render:
                    env.render()
                action, _ = agent.get_action(torch.from_numpy(state).float().to(device))
                action = action.cpu().detach().numpy()
                next_state, reward, done, info = env.step(action)
                transition = make_transition(state, \
                                             action, \
                                             np.array([reward * args.reward_scaling]), \
                                             next_state, \
                                             np.array([done]) \
                                             )
                agent.put_data(transition)

                state = next_state

                if agent.data.data_idx > agent_args.learn_start_size:
                    agent.train_net(agent_args.batch_size)

            if iteration % 5 == 0:
                for iter in range(args.iteration):

                    episode_reward = 0
                    episode_steps = 0
                    done = False
                    state = test_env[iter].reset()
                    print("total_num_step: {} , algorithm: {}  Evaluating: {}".format(total_num, i, iter))
                    while not done:
                        action, _ = agent.get_action(torch.from_numpy(state).float().to(device))
                        action = action.cpu().detach().numpy()
                        next_state, reward, done, info = env.step(action)
                        state = next_state

                        episode_steps += 1
                        episode_reward += reward

                    reward_lst[i].append(episode_reward)
                    num_lst[i].append(total_num)

#for csv
name_list=["D2RL {} num_layer:5".format(args.algo),"{} num_layer:4".format(args.algo),"{} num_layer:5".format(args.algo),"{} num_layer:6".format(args.algo)
    ,"{} num_layer:8".format(args.algo)]
csv_algo_list=list()
csv_inter_list=list()
csv_reward_list=list()

for i in range(5):
    name=[name_list[i] for n in range(len(num_lst[i]))]
    csv_algo_list.extend(name)
    csv_inter_list.extend(num_lst[i])
    csv_reward_list.extend(reward_lst[i])



my_df=pd.DataFrame({"Model":csv_algo_list,"Interaction":csv_inter_list,"Accumulated Reward":csv_reward_list})
plt.figure(figsize=(30,10))
plt.title(args.env_name)
sns.lineplot(x="Interaction",y="Accumulated Reward",hue="Model",data=my_df)
plt.savefig('{}.png'.format(args.env_name))
