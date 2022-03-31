import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
from my_env import RoomMap, NUM_INPUTS, NUM_OUTPUTS
import time
import os 
import random

MAP_PATH = 'map3.png'

# hyperparameters
hidden_size = 512
learning_rate = 0.001

# Constants
GAMMA = 0.99
num_steps = 510
max_episodes = 300000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(ActorCritic, self).__init__()

        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        # self.critic_linear2 = nn.Linear(hidden_size, hidden_size)
        self.critic_linear3 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        # self.actor_linear2 = nn.Linear(hidden_size, hidden_size)
        self.actor_linear3 = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        
        state = Variable(torch.tensor(state).unsqueeze(0).to(device))
        value = F.relu(self.critic_linear1(state))
        # value = F.relu(self.critic_linear2(value))
        value = self.critic_linear3(value)
        
        policy_dist = F.relu(self.actor_linear1(state))
        # policy_dist = self.actor_linear2(policy_dist)
        # policy_dist = F.relu(self.actor_linear2(policy_dist))
        policy_dist  = self.actor_linear3(policy_dist)
        # print(policy_dist)
        policy_dist = F.softmax(policy_dist, dim=1)

        return value, policy_dist

    def save(self, file_name = 'model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

def a2c(env):
    
    num_inputs = NUM_INPUTS
    num_outputs = NUM_OUTPUTS

    actor_critic = ActorCritic(num_inputs, num_outputs, hidden_size)
    actor_critic.load_state_dict(torch.load('model/model_6_backup.pth', map_location=torch.device(device)))
    actor_critic = actor_critic.to(device)
    ac_optimizer = optim.Adam(actor_critic.parameters(), lr=learning_rate)

    all_lengths = []
    average_lengths = []
    all_rewards = []
    entropy_term = 0

    for episode in range(max_episodes):
        log_probs = []
        values = []
        rewards = []

        # dt = 0
        # lasttime = time.time()
        state = env.reset()
        
        for steps in range(num_steps):
            
            # print(state)
            value, policy_dist = actor_critic.forward(state)
            # print(policy_dist)
            value = value.detach()[0,0]
            dist = policy_dist.detach()
            
            action = torch.argmax(policy_dist).item()
            # print(action)
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            
            entropy = -torch.sum(torch.mean(dist) * torch.log(dist))
            # dt = time.time() - lasttime
            new_state, reward, done = env.step(action)

            # lasttime = time.time()

            rewards.append(reward)
            values.append(value)

            log_probs.append(log_prob)
            entropy_term += entropy
            state = new_state
            env.render()
        
            if done or steps == num_steps-1:
                Qval, _ = actor_critic.forward(state)
                
                Qval = Qval.detach()[0,0]
                
                all_rewards.append(np.sum(rewards))
                all_lengths.append(steps)

                print('Episode {} with reward = {}'.format(episode, torch.sum(torch.tensor(rewards))))
                average_lengths.append(torch.mean(torch.tensor(all_lengths[-10:], dtype=torch.float16)))
                # if episode % 10 == 0:                    
                #     sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode, torch.sum(torch.tensor(rewards)), steps, average_lengths[-1]))
                break
            

        # compute Q values
        Qvals = torch.zeros_like(torch.tensor(values))
        
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval

        #update actor critic
        values = torch.FloatTensor(values)
        Qvals = torch.FloatTensor(Qvals)
        log_probs = torch.stack(log_probs)
        
        log_probs = log_probs.to(device)

        advantage = Qvals - values
        advantage = advantage.to(device)
        
        actor_loss = (-log_probs * advantage).mean()
        
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss + 0.0001 * entropy_term

        # print('ac_loss = ', ac_loss)
        # print('actor_loss = ', actor_loss)
        # print('critic_loss = ', critic_loss)
        # print('entropy = ', entropy_term)
        
        ac_optimizer.zero_grad()
        ac_loss.backward()
        ac_optimizer.step()

        if(episode % 100 == 0 and episode!=0):
          print("Saved model after {} games".format(episode))
        #   actor_critic.save(file_name='model_1_backup.pth')

        
    '''
    # Plot results
    smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
    smoothed_rewards = [elem for elem in smoothed_rewards]
    plt.plot(all_rewards)
    plt.plot(smoothend_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    plt.plot(all_lengths)
    plt.plot(average_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.show()
    '''

if __name__ == "__main__":
    env = RoomMap(MAP_PATH)
    # env.render()
    a2c(env)  