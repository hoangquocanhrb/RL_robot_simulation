import gym
import torch as T
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.autograd import Variable
from my_env import RoomMap, NUM_INPUTS, NUM_OUTPUTS
import os 
import time 

MAP_PATH_1 = 'map1.png'
MAP_PATH_2 = 'map1.png'

class SharedAdam(T.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
            weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = T.zeros_like(p.data)
                state['exp_avg_sq'] = T.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions, gamma=0.99):
        super(ActorCritic, self).__init__()

        self.gamma = gamma

        # self.pi1 = nn.Linear(input_dims, 256)
        # self.v1 = nn.Linear(input_dims, 256)
        # self.pi = nn.Linear(256, n_actions)
        # self.v = nn.Linear(256, 1)

        # self.rewards = []
        # self.actions = []
        # self.states = []
        self.critic_linear1 = nn.Linear(input_dims, 512)
        self.critic_linear3 = nn.Linear(512, 1)

        self.actor_linear1 = nn.Linear(input_dims, 512)
        self.actor_linear3 = nn.Linear(512, n_actions)
    
    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def forward(self, state):
        # pi1 = F.relu(self.pi1(state))
        # v1 = F.relu(self.v1(state))

        # pi = F.softmax(self.pi(pi1), dim=1)
        # v = self.v(v1)

        # return pi, v
        # state = Variable(T.tensor(state).unsqueeze(0))
        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear3(value)
        
        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear3(policy_dist), dim=1)

        return policy_dist, value

    def calc_R(self, done):
        states = T.tensor(self.states, dtype=T.float)
        _, v = self.forward(states)

        R = v[-1]*(1-int(done))

        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma*R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = T.tensor(batch_return, dtype=T.float)

        return batch_return

    def calc_loss(self, done):
        states = T.tensor(self.states, dtype=T.float)
        actions = T.tensor(self.actions, dtype=T.float)

        returns = self.calc_R(done)

        pi, values = self.forward(states)
        values = values.squeeze()
        critic_loss = (returns-values)**2

        probs = T.softmax(pi, dim=1)
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs*(returns-values)

        total_loss = (critic_loss + actor_loss).mean()
    
        return total_loss

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float)
        pi, v = self.forward(state)
        # probs = T.softmax(pi, dim=1)
        # dist = Categorical(pi)
        # action = dist.sample().numpy()[0]
        
        dist = pi.detach()
        action = T.argmax(dist).item()
        return action

    def save(self, file_name = 'model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        T.save(self.state_dict(), file_name)

class Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions, 
                gamma, lr, name, global_ep_idx, env_id):
        super(Agent, self).__init__()
        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma)
        self.global_actor_critic = global_actor_critic
        self.name = 'Agent {}'.format(name)
        self.episode_idx = global_ep_idx
        self.env = env_id
        self.optimizer = optimizer
        self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())
        self.local_actor_critic.eval()

    def run(self):
        t_step = 1
        while self.episode_idx.value < N_GAMES:
            done = False
            observation = self.env.reset()
            score = 0
            self.local_actor_critic.clear_memory()

            while not done:
                action = self.local_actor_critic.choose_action(observation)

                observation_, reward, done = self.env.step(action)
                score += reward
                self.local_actor_critic.remember(observation, action, reward)
                lasttime = time.time() # update lasttime

                # if t_step % T_MAX == 0 or done:
                #     loss = self.local_actor_critic.calc_loss(done)
                #     self.optimizer.zero_grad()
                #     loss.backward()
                #     for local_param, global_param in zip(
                #             self.local_actor_critic.parameters(),
                #             self.global_actor_critic.parameters()):
                #         global_param._grad = local_param.grad
                #     self.optimizer.step()
                #     self.local_actor_critic.load_state_dict(
                #             self.global_actor_critic.state_dict())
                #     self.local_actor_critic.clear_memory()
                # t_step += 1
                self.env.render()
                observation = observation_

            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
            print(self.name, 'episode ', self.episode_idx.value, 'reward %.4f' % score)
            # if(self.episode_idx.value % 100 ==0 and self.episode_idx.value != 0):
                # self.global_actor_critic.save(file_name='A3C_2_backup.pth')
            
            

if __name__ == '__main__':
    lr = 0.0003
    env_id = [RoomMap(MAP_PATH_1), RoomMap(MAP_PATH_2)]
    n_actions = NUM_OUTPUTS
    input_dims = NUM_INPUTS
    N_GAMES = 1000001
    T_MAX = 100
    global_actor_critic = ActorCritic(input_dims, n_actions)
    global_actor_critic.load_state_dict(T.load('model/model_6_backup.pth', map_location=T.device('cpu')))
    
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr, 
                        betas=(0.92, 0.999))
    global_ep = mp.Value('i', 0)

    workers = [Agent(global_actor_critic,
                    optim,
                    input_dims,
                    n_actions,
                    gamma=0.99,
                    lr=lr,
                    name=i,
                    global_ep_idx=global_ep,
                    env_id=env_id[i]) for i in range(2)] #mp.cpu_count()
    [w.start() for w in workers]
    [w.join() for w in workers]
