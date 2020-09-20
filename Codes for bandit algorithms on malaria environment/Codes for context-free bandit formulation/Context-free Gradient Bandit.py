#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#from tqdm import trange
#matplotlib.use('TkAgg')
import gym 
import multiprocessing as mp
import os
import argparse
import random
import json


# In[7]:


class Agent(object):
    """
    An Agent is able to take one of a set of actions at each time step. The
    action is chosen using a strategy based on the history of prior actions
    and outcome observations.
    """
    def __init__(self,environment,k_arms, policy,initial=0.,prior=0,alpha=None,baseline=False):
        self.policy = policy
        self.environment = environment
        self.k_arms = k_arms 
        self.indices = np.arange(self.k_arms)
        self.prior = prior
        self.baseline =baseline
        self.initial=initial
        self._value_estimates = self.prior*np.ones(self.k_arms)
        if self.initial>0:
            self._value_estimates = np.zeros(self.k_arms) + self.initial
        self.action_attempts = np.zeros(self.k_arms)
        self.t = 0
        self.average_reward=0
        self.last_action = None

    def __str__(self):
        return 'f/{}'.format(str(self.policy))

    def reset(self):
        """
        Resets the agent's memory to an initial state.
        """
        self._value_estimates[:] = self.prior + self.initial
        self.action_attempts[:] = 0
        self.last_action = None
        self.t = 0

    def choose(self, evaluate=False):
        action = self.policy.choose(self)
        self.last_action = action
        return action

    def observe(self, reward):
        self.t += 1
        self.average_reward += (reward - self.average_reward) / self.t
        self.action_attempts[self.last_action] += 1

        if self.gamma is None:
            g = 1 / self.action_attempts[self.last_action]
        else:
            g = self.gamma
        if self.gradient:
            one_hot = np.zeros(self.k_arms)
            one_hot[self.last_action] = 1
            if self.gradient_baseline:
                baseline = self.average_reward
            else:
                baseline = 0
            self._value_estimates += self.gamma * (reward - baseline) * (one_hot - self.policy.action_prob) # stochastic gradient ascent
        else:
            q = self._value_estimates[self.last_action]
            self._value_estimates[self.last_action] += g*(reward - q)
        

    @property
    def value_estimates(self):
        return self._value_estimates
class GradientAgent(Agent):
    """
    The Gradient Agent learns the relative difference between actions instead of
    determining estimates of reward values. It effectively learns a preference
    for one action over another.
    """
    def __init__(self, environment,k_arms,policy,initial=0, prior=0, alpha=None, baseline=True):
        super(GradientAgent, self).__init__(environment,k_arms, policy, prior,alpha)
        self.alpha = alpha
       
        self.baseline = baseline
        self.average_reward = 0

    def __str__(self):
        return 'g/\u03B1={}, bl={}'.format(self.alpha, self.baseline)

    def observe(self, reward):
        self.action_attempts[self.last_action] += 1

        if self.baseline:
            diff = reward - self.average_reward
            self.average_reward += 1/np.sum(self.action_attempts) * diff

        pi = np.exp(self.value_estimates) / np.sum(np.exp(self.value_estimates))

        ht = self.value_estimates[self.last_action]
        ht += self.alpha*(reward - self.average_reward)*(1-pi[self.last_action])
        self._value_estimates -= self.alpha*(reward - self.average_reward)*pi
        self._value_estimates[self.last_action] = ht
        self.t += 1

    def reset(self):
        super(GradientAgent, self).reset()
        self.average_reward = 0



class Policy(object):
    """
    A policy prescribes an action to be taken based on the memory of an agent.
    """
    def __str__(self):
        return 'generic policy'

    def choose(self, agent, evaluate=False):
        return 0
class SoftmaxPolicy(Policy):
    """
    The Softmax policy converts the estimated arm rewards into probabilities
    then randomly samples from the resultant distribution. This policy is
    primarily employed by the Gradient Agent for learning relative preferences.
    """
    def __str__(self):
        return 'SM'

    def choose(self, agent):
        a = agent.value_estimates
        pi = np.exp(a) / np.sum(np.exp(a))
        print("pi",pi)
        cdf = np.cumsum(pi)
        print("cdf=",cdf)
        s = np.random.random()
        print("s=",s)
        return np.where(s < cdf)[0][0]


# In[8]:


class MalariaAgent:
    def __init__(self, environment, policy,initial=0,
                 prior=0,alpha=None, 
                 baseline=False,
                 n_episodes=10):
        self.env = environment
        self.action_space = environment.action_space #List of 2-dim actions
        self.k_arms = len(self.action_space)
        self.agent = GradientAgent(environment,
                           self.k_arms,
                           policy,initial,
                           prior,alpha,baseline)
        self.n_episodes = n_episodes
        self.episodic_rewards = []
        
    def train(self):
        allrewards = []
        for episode in range(self.n_episodes):
            print("episode {}".format(episode+1))
            observation = self.env.reset()
            action = self.agent.choose()
            eps_reward = 0
            rewards = []
            while True:
                observation, reward, done, info = self.env.step(self.action_space[action])
                rewards.append(reward)
                eps_reward+=reward
                if done:
                    print("Episode {} finished!".format(episode+1))
                    print('episodic reward: ',sum(rewards))
                    break
            self.agent.observe(sum(rewards))
            allrewards.append(sum(rewards))
        
        self.episodic_rewards = allrewards
        return allrewards
    
    def evaluate(self):
        observation = self.env.reset()
        action = self.agent.choose(evaluate=True)
        eps_reward = 0
        rewards = []
        policy = []
        while True:
            observation, reward, done, info = self.env.step(self.action_space[action])
            rewards.append(reward)
            eps_reward+=reward
            policy.append(self.action_space[action])
            if done:
                print('episodic reward: ',sum(rewards))
                break
                
        return policy, sum(rewards)
    
    def generate(self):
        print('Training policy...')
        self.train()
        print('Evaluating policy...')
        best_policy, best_reward = self.evaluate()
        return best_policy, best_reward




def train_agent(agent):
    np.random.seed(random.randint(0,142342))
    policy, reward = agent.generate()
    return agent.episodic_rewards, policy, reward

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("num_exp", type=int,
                        help="number of experiments per agent type")
    parser.add_argument("num_episodes", type=int,
                        help="number of episodes")
    args = parser.parse_args()
    
    num_exp = args.num_exp
    num_episodes = args.num_episodes 
    alpha =[0.01,0.1,0.01,0.1]
    #gamma =[0.01]
    options=[True,True,False,False]
    num_cores = os.cpu_count()
    
    data = dict()
    labels = []
    x=np.arange(num_episodes)
    for i, alp in enumerate(alpha):
        agent_id = 'agent_'+str(i)
        data[agent_id] = {'alpha':alp}
        if i<2:
            labels.append('alpha_{}= {} {} '.format(i+1,alp,"with baseline"))
        else:
            labels.append('alpha_{}= {} {} '.format(i+1,alp,"without baseline"))
        agents = [MalariaAgent(environment=gym.make("ushiriki_policy_engine_library:ChallengeAction-v0", userID="61122946-1832-11ea-8d71-362b9e155667"),
                  policy=SoftmaxPolicy(),initial=0,prior=0,alpha=alp,baseline=options[i],
                               n_episodes=num_episodes) for _ in range(num_exp)]
        
        with mp.Pool(processes=num_cores) as pool:
            results = pool.map(train_agent, agents)
        
        train_rewards = []
        policies = []
        rewards = []
        for result in results:
            train_rewards.append(result[0])
            policies.append(result[1])
            rewards.append(result[2])
        data[agent_id]['training'] = train_rewards
        data[agent_id]['evaluation'] = {'policies':policies, 'rewards':rewards}
        
        plt.plot(x, np.mean(np.array(train_rewards),axis=0), label=labels[i])
        plt.fill_between(x, 
                 np.mean(np.array(train_rewards),axis=0)-np.std(np.array(train_rewards),axis=0), 
                 np.mean(np.array(train_rewards),axis=0)+np.std(np.array(train_rewards),axis=0),
                 alpha=0.3)
        
    #plt.title("Plot for GradientBandit(context-free)")
    plt.xlabel('episodes')
    plt.ylabel('average episodic reward')
    plt.legend(shadow=True,fancybox=True,framealpha=0.7,loc="best")
    plt.savefig('GradientBandit_contextfree.png')
    
    full_json_dir = "GradientBandit_contextfree.json"
    with open(full_json_dir, 'w') as outfile:
        json.dump(data, outfile)


# In[ ]:




