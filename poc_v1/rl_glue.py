#!/usr/bin/env python

"""Glues together an experiment, agent, and environment.
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import numpy as np
from matplotlib.animation import FuncAnimation


class RLGlue:
    """RLGlue class
    
    General components:
    - Starts the environment and agents
    - Calls rl_episode to step through an episode
        IF (not terminal) THEN rl_step:
            a) Take action selected and do an environment step
            b) Figure out the next action
            c) update step count and reward total
    - Tracks metadata of the episode(s) and run

    args:
        env_name (string): the name of the module where the Environment class can be found
        agent_name (string): the name of the module where the Agent class can be found
    outputs:
        total return per episode
        number of steps per episode
        number of episodes
        plots
    """

    def __init__(self, env_class, agent_class):
        self.environment = env_class()
        self.agent = agent_class()

        self.total_reward = None
        self.last_action = None
        self.num_steps = None
        self.num_episodes = None
        
        self.animate = False

    def rl_init(self, agent_init_info={}, env_init_info={}):
        """Initial method called when RLGlue experiment is created"""
        self.environment.env_init(env_init_info)
        self.agent.agent_init(agent_init_info)

        self.total_reward = 0.0
        self.num_steps = 0
        self.num_episodes = 0

    def rl_start(self, agent_start_info={}, env_start_info={}):
        """Starts RLGlue experiment

        Returns:
            tuple: (state, action)
        """

        last_state = self.environment.env_start()
        self.last_action = self.agent.agent_start(last_state)

        observation = (last_state, self.last_action)

        return observation

    def rl_agent_start(self, observation):
        """Starts the agent.

        Args:
            observation: The first observation from the environment

        Returns:
            The action taken by the agent.
        """
        return self.agent.agent_start(observation)

    def rl_agent_step(self, reward, observation):
        """Step taken by the agent

        Args:
            reward (float): the last reward the agent received for taking the
                last action.
            observation : the state observation the agent receives from the
                environment.

        Returns:
            The action taken by the agent.
        """
        return self.agent.agent_step(reward, observation)

    def rl_agent_end(self, reward):
        """Run when the agent terminates

        Args:
            reward (float): the reward the agent received when terminating
        """
        self.agent.agent_end(reward)

    def rl_env_start(self):
        """Starts RL-Glue environment.

        Returns:
            (float, state, Boolean): reward, state observation, boolean
                indicating termination
        """
        self.total_reward = 0.0
        self.num_steps = 1

        this_observation = self.environment.env_start()

        return this_observation

    '''
    def rl_env_step(self, action):
        """Step taken by the environment based on action from agent
        
        --- USUSED??
        
        Args:
            action: Action taken by agent.

        Returns:
            (float, state, Boolean): reward, state observation, boolean
                indicating termination.
        """
        ro = self.environment.env_step(action)
        (this_reward, _, terminal) = ro

        self.total_reward += this_reward

        if terminal:
            self.num_episodes += 1
        else:
            self.num_steps += 1

        return ro
    '''

    def rl_step(self):
        """Step taken by RLGlue, takes environment step and either step or
            end by agent.
            
        Local_Args:
            step_by_step (Bol): decides whether you do the RL Step by Step for debugging purposes
            
        Returns:
            (float, state, action, Boolean): reward, last state observation,
                last action, boolean indicating termination
        """
        step_by_step = self.agent.step_by_step
        
        (reward, last_state, term) = self.environment.env_step(self.last_action)

        # Reward for the episode
        self.total_reward += reward;

        if term:
            self.num_episodes += 1
            if step_by_step:
                self.print_rl_step(reward, last_state, term, "None")
                input("Press ENTER to continue.")
            else:
                pass
            self.agent.agent_end(reward)
            roat = (reward, last_state, None, term)
        else:
            self.num_steps += 1
            next_action = self.agent.agent_step(reward, last_state)
            
            # Prints or animates step-by-step
            if self.animate:
                self.print_rl_step(reward, last_state, term, next_action)
            elif step_by_step:   
                self.print_rl_step(reward, last_state, term, next_action)
                input("Press ENTER to continue.")
            else:
                pass
            
            self.last_action = next_action
            roat = (reward, last_state, self.last_action, term)

        return roat

    def rl_cleanup(self):
        """Cleanup done at end of experiment."""
        self.environment.env_cleanup()
        self.agent.agent_cleanup()

    def rl_agent_message(self, message):
        """Message passed to communicate with agent during experiment

        Args:
            message: the message (or question) to send to the agent

        Returns:
            The message back (or answer) from the agent

        """

        return self.agent.agent_message(message)

    def rl_env_message(self, message):
        """Message passed to communicate with environment during experiment

        Args:
            message: the message (or question) to send to the environment

        Returns:
            The message back (or answer) from the environment

        """
        return self.environment.env_message(message)

    def rl_episode(self, max_steps_this_episode):
        """Runs an RLGlue episode

        Args:
            max_steps_this_episode (Int): the maximum steps for the experiment to run in an episode

        Returns:
            Boolean: if the episode should terminate
        """
        is_terminal = False
        
        self.rl_start()
        self.num_steps = 0 # RESET NUM STEPS TO 0 AT START OF EPISODE

        while (not is_terminal) and ((max_steps_this_episode == 0) or
                                     (self.num_steps < max_steps_this_episode)):
            rl_step_result = self.rl_step()
            is_terminal = rl_step_result[3]

        return is_terminal

    def rl_return(self):
        """The total reward

        Returns:
            float: the total reward for the episode
        """
        return self.total_reward

    def rl_num_steps(self):
        """The total number of steps taken

        Returns:
            Int: the total number of steps taken
        """
        return self.num_steps

    def rl_num_episodes(self):
        """The number of episodes

        Returns
            Int: the total number of episodes

        """
        return self.num_episodes
   
    def env_print_state(self, next_action):
        
        # Clear plot
        plt.clf()
        
        dim = int(np.sqrt(len(self.environment.state[1])))
        grid = self.environment.state[1].reshape((dim,dim), order = "C")

        # determine who owns which territories
        player_squares = np.zeros(len(self.environment.state[1]))
        for i in range(len(player_squares)):
            if self.environment.state[1][i] > 0:
                player_squares[i] = 1
            else:
                player_squares[i] = 0
        grid_player = player_squares.reshape((dim,dim), order = "C")
        
        cmap = ListedColormap(['red', 'blue']) # player colors
        ax = sns.heatmap(grid_player, cmap=cmap, annot=grid, cbar=False)
        
        # Plot the attack direction
        # get the attack array representation
        from_territory, to_territory, terminal = self.environment.action_map[next_action]
        # convert the array representation to x, y coords
        dim = int(self.environment.dim) # This could be optimized
        y_1, x_1 = np.unravel_index(int(from_territory), (dim, dim)) # For some reason I need to flip the x, y to make sense
        y_2, x_2 = np.unravel_index(int(to_territory), (dim, dim)) 
        # plot        
        plt.arrow((x_1+0.65), (y_1+0.4), (x_2-x_1), (y_2-y_1), width = 0.05, color = 'black')
        
        #plot the borders
        for i in range(grid.shape[1]+1):
            #if (i == int(dim/2+1)) or (i == int(dim/2)):  # for dim>4
            if (i == int(dim/2)):  
                ax.axvline(i, color='white', lw=2)
                ax.axhline(i, color='white', lw=2)

        plt.xticks(np.arange(0, dim+1, 1.0))
        plt.yticks(np.arange(0, dim+1, 1.0))
        plt.title("Animation of RL Agent for Risk - Step Num: " + str(self.num_steps))
         
        # Save the images for later animation processing
        if self.animate == True:
            plt.savefig("animation_pics/num_step_"+str(self.num_steps)+".jpg", dpi = 300)
        else:
            plt.show()
        
    def animation(self):
        self.animate = True
 
    def print_rl_step(self, reward, last_state, term, next_action):
        
        if self.animate == True:
            self.env_print_state(next_action)
        else:  
            print("\nRL Step Number: " + str(self.num_steps) + " of Episode Number: " + str(self.num_episodes))
            print("\nLast Action: " + str(self.last_action) + " resulting in state below and Reward: " + str(reward) + " Total Reward of: " + str(self.total_reward) + " and Terminal status of: " + str(term))
            self.env_print_state(next_action)
            print("\nNext Action: " + str(next_action))
        
        
