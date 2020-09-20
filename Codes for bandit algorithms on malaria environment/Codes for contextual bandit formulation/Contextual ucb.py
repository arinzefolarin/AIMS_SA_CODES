#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python


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
#import pymc3 as pm


class Agent(object):
    """
    An Agent is able to take one of a set of actions at each time step. The
    action is chosen using a strategy based on the history of prior actions
    and outcome observations.
    """
    def __init__(self,environment,k_arms, policy,initial=0.,prior=0, gamma=None,gradient=False, gradient_baseline=False):
        self.policy = policy
        self.environment = environment
        self.k_arms = k_arms 
        self.prior = prior
        self.gamma = gamma
        self.gradient = gradient
        self.gradient_baseline =gradient_baseline
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
            self._value_estimates += self.gamma * (reward - baseline) * (one_hot - self.policy.pi) # stochastic gradient ascent
        else:
            q = self._value_estimates[self.last_action]
            self._value_estimates[self.last_action] += g*(reward - q)
        

    @property
    def value_estimates(self):
        return self._value_estimates


class Policy(object):
    """
    A policy prescribes an action to be taken based on the memory of an agent.
    """
    def __str__(self):
        return 'generic policy'

    def choose(self, agent, evaluate=False):
        return 0


class EpsilonGreedyPolicy(Policy):
    """
    The Epsilon-Greedy policy will choose a random action with probability
    epsilon and take the best apparent approach with probability 1-epsilon. If
    multiple actions are tied for best choice, then a random action from that
    subset is selected.
    """
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __str__(self):
        return '\u03B5-greedy (\u03B5={})'.format(self.epsilon)

    def choose(self, agent, evaluate=False):
        if evaluate:
            action = np.argmax(agent.value_estimates)
            check = np.where(agent.value_estimates == agent.value_estimates[action])[0]
            if len(check) == 1:
                return action
            else:
                return np.random.choice(check)
        else:
            if np.random.random() < self.epsilon:
                return np.random.choice(len(agent.value_estimates))
            else:
                action = np.argmax(agent.value_estimates)
                check = np.where(agent.value_estimates == agent.value_estimates[action])[0]
                if len(check) == 1:
                    return action
                else:
                    return np.random.choice(check)


class DecayingEpsilonGreedyPolicy(Policy):
    """
    The Epsilon-Greedy policy will choose a random action with probability
    epsilon and take the best apparent approach with probability 1-epsilon. If
    multiple actions are tied for best choice, then a random action from that
    subset is selected.
    """
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def __str__(self):
        return '\u03B5-greedy (\u03B5={})'.format(self.epsilon)

    def choose(self, agent, evaluate=False):
        if evaluate:
            action = np.argmax(agent.value_estimates)
            check = np.where(agent.value_estimates == agent.value_estimates[action])[0]
            if len(check) == 1:
                return action
            else:
                return np.random.choice(check)
        else:
            if np.random.random() < self.epsilon:
                self.epsilon = self.epsilon*0.9
                print(self.epsilon)
                return np.random.choice(len(agent.value_estimates))
            else:
                action = np.argmax(agent.value_estimates)
                check = np.where(agent.value_estimates == agent.value_estimates[action])[0]
                if len(check) == 1:
                    return action
                else:
                    return np.random.choice(check)

            
class DecayingEpsilonGreedyPolicy1(Policy):
    """
    The Epsilon-Greedy policy will choose a random action with probability
    epsilon and take the best apparent approach with probability 1-epsilon. If
    multiple actions are tied for best choice, then a random action from that
    subset is selected.
    """
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.time=0

    def __str__(self):
        return '\u03B5-greedy (\u03B5={})'.format(self.epsilon)

    def choose(self, agent, evaluate=False):
        if evaluate:
            action = np.argmax(agent.value_estimates)
            check = np.where(agent.value_estimates == agent.value_estimates[action])[0]
            if len(check) == 1:
                return action
            else:
                return np.random.choice(check)
        else:
            self.time += 1
            print(self.time)
            self.epsilon = 1/np.log(self.time+0.00001)
            if np.random.random() < self.epsilon:
                print(self.epsilon)
                return np.random.choice(len(agent.value_estimates))
            else:
                action = np.argmax(agent.value_estimates)
                check = np.where(agent.value_estimates == agent.value_estimates[action])[0]
                if len(check) == 1:
                    return action
                else:
                    return np.random.choice(check)
        
class GreedyPolicy(EpsilonGreedyPolicy):
    """
    The Greedy policy only takes the best apparent action, with ties broken by
    random selection. This can be seen as a special case of EpsilonGreedy where
    epsilon = 0 i.e. always exploit.
    """
    def __init__(self):
        super(GreedyPolicy, self).__init__(0)

    def __str__(self):
        return 'greedy'


class RandomPolicy(EpsilonGreedyPolicy):
    """
    The Random policy randomly selects from all available actions with no
    consideration to which is apparently best. This can be seen as a special
    case of EpsilonGreedy where epsilon = 1 i.e. always explore.
    """
    def __init__(self):
        super(RandomPolicy, self).__init__(1)

    def __str__(self):
        return 'random'


class UCBPolicy(Policy):
    """
    The Upper Confidence Bound algorithm (UCB1). It applies an exploration
    factor to the expected value of each arm which can influence a greedy
    selection strategy to more intelligently explore less confident options.
    """
    def __init__(self, c):
        self.c = c

    def __str__(self):
        return 'UCB (c={})'.format(self.c)

    def choose(self, agent, evaluate=False):
        if evaluate:
            q = agent.value_estimates
            action = np.argmax(q)
            check = np.where(q == q[action])[0]
            if len(check) == 1:
                return action
            else:
                return np.random.choice(check)
        else:
            UCB_estimation = agent.value_estimates + \
            self.c * np.sqrt(np.log(agent.t + 1) / (agent.action_attempts + 1e-5))
            q_best = np.max(UCB_estimation) ## The UCB formular
            return np.random.choice(np.where(UCB_estimation == q_best)[0]) ## Returns the action


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
        self.pi = np.exp(a) / np.sum(np.exp(a))
        cdf = np.cumsum(self.pi)
        s = np.random.random()
        return np.where(s < cdf)[0][0]
class GradientBanditPolicy(Policy):
    def __str__(self):
        return 'GB'

    def choose(self, agent,evaluate=False):
        if evaluate:
            q = agent.value_estimates
            action = np.argmax(q)
            check = np.where(q == q[action])[0]
            if len(check) == 1:
                return action
            else:
                return np.random.choice(check)
        else:
            exp_est = np.exp(agent.value_estimates) ## numerical preference exp^(H_t(a))
            self.action_prob = exp_est / np.sum(exp_est) ## Probability of taking action a at time t; pi_t(a)
            return np.random.choice(agent.indices, p=self.action_prob) ## ret




class ContextualMalariaAgent:
    def __init__(self, environment, policy,initial=0,
                 prior=0,gamma=None,gradient=False, 
                 gradient_baseline=False,
                 n_episodes=10):
        self.env = environment
        self.action_space = environment.action_space #List of 2-dim actions
        self.k_arms = len(self.action_space)
        self.agent =[ Agent(environment,
                           self.k_arms,
                           policy,initial,
                           prior,gamma) for _ in range(5)]
        self.n_episodes = n_episodes
        self.episodic_rewards = []
        
    def train(self):
        allrewards = []
        for episode in range(self.n_episodes):
            print("episode {}".format(episode+1))
            observation = self.env.reset()
            eps_reward = 0
            rewards = []
            count = 0
            while True:
                action = self.agent[count].choose()
                observation, reward, done, info = self.env.step(self.action_space[action])
                self.agent[count].observe(reward)
                rewards.append(reward)
                eps_reward+=reward
                count +=1
                if done:
                    print("Episode {} finished!".format(episode+1))
                    print('episodic reward: ',sum(rewards))
                    break
            allrewards.append(sum(rewards))
        
        self.episodic_rewards = allrewards
        return allrewards
    
    def evaluate(self):
        observation = self.env.reset()
        eps_reward = 0
        count = 0
        rewards = []
        policy = []
        while True:
            action = self.agent[count].choose(evaluate=True)
            observation, reward, done, info = self.env.step(self.action_space[action])
            rewards.append(reward)
            eps_reward+=reward
            policy.append(self.action_space[action])
            count+=1
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
    UCB_params =[0.5,1.0,2.0]
    num_cores = os.cpu_count()
    
    data = dict()
    labels = []
    x=np.arange(num_episodes)
    for i, ucb in enumerate(UCB_params):
        agent_id = 'agent_'+str(i)
        data[agent_id] = {'ucb_param':ucb} 
        labels.append('ucb='+str(ucb))
        agents = [ContextualMalariaAgent(environment=gym.make("ushiriki_policy_engine_library:ChallengeAction-v0", userID="61122946-1832-11ea-8d71-362b9e155667"),
                  policy=UCBPolicy(ucb),gamma=0.9,initial=0,prior=0, 
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
        
    plt.title("Training plot for UCB (Contextual)")
    plt.xlabel('episodes')
    plt.ylabel('average episodic reward')
    plt.legend(shadow=True,fancybox=True,framealpha=0.7,loc="best")
    #plt.show()
    plt.savefig('contextualUCB.png')
    
    full_json_dir = "contextual_epsilon_greedy_UCB.json"
    with open(full_json_dir, 'w') as outfile:
        json.dump(data, outfile)







