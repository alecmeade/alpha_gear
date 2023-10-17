from Coursera_lab_files import environment
from dynamics import *
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

class RiskEnvironmentPOC(environment.BaseEnvironment):
    def __init__(self):
        self.current_state = None
        self.count = 0
        
    def env_init(self, env_info):
        """Setup for the environment called when the experiment first starts.

        Note:
            Initialize a tuple with the reward, first state observation, boolean
            indicating if it's terminal.
            
            Initializes the action_map
        """
        # number of states - should be a perfect square for this setup
        self.state = [0, np.zeros(env_info['states']), False] #reward, state init, boolean
        dim = env_info['states']**0.5
        self.dim = dim #note - this could be optimized
        #print("dim is: " + str(dim))
        
        # number of actions
        self.num_actions = env_info["num_actions"]
        
        # initialize the action_map
        """ 
        from action_num, determine the territory [from , to, itself?]
        """
        self.action_map = {}

        '''
        E.g. with state dimension of 25...
        [0] [1] [2] [3] [4]
        [5] [6] [7] [8] [9]
        [10][11][12][13][14]
        [15][16][17][18][19]
        [20][21][22][23][24]
        '''
        # [From, To, out_of_bounds]
        if self.num_actions == 4:
            for i in range(env_info['states']):
                # each territory has 4 actions from each state: up, down, left, right & 1 action which is do nothing
                # if out of bounds, then it doesn't do anything
                self.action_map[i*self.num_actions] = [i, i-dim, False] if i-dim >= 0 else [i, i, True] # up
                self.action_map[i*self.num_actions + 1] = [i, i+dim, False] if i+dim < env_info['states'] else [i, i, True] # down
                self.action_map[i*self.num_actions + 2] = [i, i-1, False] if (i)%dim != 0 else [i, i, True] # left
                self.action_map[i*self.num_actions + 3] = [i, i+1, False] if (i)%dim != (dim-1) else [i, i, True] # right
            self.action_map[i*self.num_actions + 4] = [i, i, True] # do nothing    

        else:
            raise ValueError("Incorrect number of actions passed")
        # print(self.action_map)
        
    
    def env_start(self):
        """The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        # Set up the board with troops
        # where 'positive = agent troops' and 'negative = neutral troops'
        states_remaining = list(range(len(self.state[1]))) # list of states to choose from
        turn = True # whether it is the turn of the agent to be assigned troops somewhere or not
        
        for i in range(len(self.state[1])):
            
            # determine index to assign troops to
            index = random.choice(states_remaining)
            states_remaining.remove(index)
            
            if turn == True:
                self.state[1][index] = 3
                turn = False
            else:
                self.state[1][index] = -3
                turn = True  

        self.current_state = self.state[1]
        return self.current_state
    
    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """
        #############
        # Dynamics
        ### functions that take in the action and return the observation
        # X = func(action)
        # initialize
        terminal = False
        reward = 0.0
        last_state = np.array(self.state[1].copy()) #need copy because self.state[1] changes after assignment
        
        observation = action_state(self.state[1], action, self.num_actions, self.action_map) # state from dynamics
        #############
        
        # use the above observations to decide what the reward will be, and if the agent is in a terminal state.
        
        #############
        # Reward function
        current_state = np.array(observation)
        
        # + Reward if a territory is captured        
        comparison_last = (np.where(current_state < 0, current_state, 0) != np.where(last_state < 0, last_state, 0)).any() 
        
        if comparison_last:
            reward += 1
        
        # Determine if terminal
        # In this simple implementation (where + numbers represent the agents territories, the game is over when all grid numbers are +ve)
        comparison_terminal = (np.where(observation < 0, observation, 0) == np.zeros(len(self.state[1]))).all() # True if no negatives

        if comparison_terminal:
            # agent wins
            terminal = True
            reward += 100
         
        else: 
            # continue
            terminal = False
            reward += -0.5 # most random games take no more than 1000 steps
        #############
        
        self.state[0] = reward
        self.state[1] = observation
        self.state[2] = terminal
        self.current_state = self.state[1]
        
        self.reward_obs_term = (reward, observation, terminal)
        return self.reward_obs_term
    
    def env_cleanup(self):
        return None
    
    def env_message(self):
        return None
    