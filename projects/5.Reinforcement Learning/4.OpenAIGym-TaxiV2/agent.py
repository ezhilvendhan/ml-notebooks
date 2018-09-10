import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon = 0.00009
        self.alpha = 0.4
        self.gamma = 1
        print('======= Parameters =======')
        print('epsilon : {}'.format(self.epsilon))
        print('alpha   : {}'.format(self.alpha))
        print('gamma   : {}'.format(self.gamma))
        print('==========================')
        
    def epsilon_greedy(self, epsilon, Q_s):
        actions = []
        max_set = False
        for _action in Q_s:
            if np.max(Q_s) == _action and not max_set:
                actions.append(1-epsilon+(epsilon/self.nA))
                max_set = True
            else:
                actions.append(epsilon/self.nA)
        return actions

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
#         return np.random.choice(self.nA)
        actions = self.epsilon_greedy(self.epsilon, self.Q[state])
        return np.random.choice(np.arange(self.nA), p=actions)
#         return np.argmax(actions)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        policy = self.epsilon_greedy(self.epsilon, self.Q[next_state])
        sum_action_state = np.dot(self.Q[next_state], policy)
#         next_action = self.select_action(next_state)
        self.Q[state][action] += self.alpha * (
            reward + (
#                 self.gamma * self.Q[next_state][next_action])
                  self.gamma * sum_action_state)
            - self.Q[state][action])