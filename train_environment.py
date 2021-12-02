import gym
import numpy as np
from numpy import array, inf
from numpy.linalg import norm


class TrainingEnv(gym.Env):
    """
    Creates a general training environment that can be used with different CFTs.
    The input to the Network are the errors for the crossing equations.
    Instructions to implement the modes as specified in arXiv:2108.09330v1:
    Mode 1: 
            Set min_state_bound and max_state_bound to the desired values. Set search_size to max_state_bound-min_state_bound.
            Leave freeze_mask set to None.
    Mode 2:
            Set min_state_bound and max_state_bound to the desired values. Set the desired search_size.
            Set freeze_mask to an array of 1s and 0s. Set freeze_accuracy to the desired value. 
            The variables in the state corresponding to the 0 entries in freeze_mask won't change during training until
            the prescribed freeze_accuracy is reached.
    Mode 3: 
            Set the initial state to the desired state. Set small search_sizes and bounds appropriately.

    """     
    def __init__(self, error_fn, accuracy_fn, state_dim, obs_dim, initial_state, 
                 min_state_bound=None, max_state_bound=None,
                 guess_mode_steps=0, trapped_steps=100, reward_cap=100, reward_scale=1.,
                 search_size=1, freeze_mask=None, freeze_accuracy=0.01):
        """
        Args:
        error_fn:        function for the crossing equations to be used for penalty.
                         Given a state as input should return the vector E.
        accuracy_fn:     function used to compute accuracy. Given a state should return ||E||/E_{abs}
        state_dim:       integer. dimension of the array of variables to be searched.
        initial state:   array of dimension state_dim. Initial values for the variables to be searched.
        min_state_bound: float or array of dimension state_dim. Sets the minimum value for the search of variables. 
                         If a float, the bound is common for all variables.
        max_state_bound: float or array of dimension state_dim. Sets the maximum value for the search of variables. 
                         If a float, the bound is common for all variables.
        guess_mode_steps: integer. Number of steps to be taken in guess mode.
        trapped_steps:   integer. Number of steps without improvement on the best state after which the environment is reset. 
        reward_cap:      float. Resets the environment when the agent finds a state with reward<reward_cap*best_reward. Should increase stability.
        reward_scale:    float. Reward is computed as -reward_scale*||E||
        search_size:      float or array of dimension state_dim. Sets the search window at each step for the variables.
                         If a float, the guess size is the same for all variables.
        freeze_mask:     array of dimension state_dim containing only 0 and 1. The variables corresponding to indices that are 0 are not changed
                         until the accuracy reach the level freeze_accuracy. This is needed for training in mode 2. If None no variable is frozen,
        freeze_accuracy: float. Value of accuracy at which all the variables are unfrozen if freeze_mask is not None
        """

        self.error_fn=error_fn
        self.accuracy_fn=accuracy_fn
        self.state_dim=state_dim
        self.obs_dim=obs_dim
        self.initial_state=initial_state
        self.state_min=min_state_bound 
        self.state_max=max_state_bound 
        self.guess_mode_steps=guess_mode_steps
        self.trapped_steps=trapped_steps 
        self.reward_cap=reward_cap 
        self.scale=reward_scale 
        self.search_size=search_size 
        if freeze_mask is not None:   #Setup freeze mode if a mask is passed
                self.freeze_mode=True      
                self.freeze_mask=freeze_mask
                self.freeze_accuracy=freeze_accuracy
        else:
            self.freeze_mode=False    

        self.step_counter=0 #Records the number of steps executed in the environment
        self.not_improving_steps=0 #Stores the number of steps since last improvement
        self.guess_mode=True #Controls wheter the algorithm is working in guess mode
        self.best_reward=None  #Records the best reward found so far
        self.best_state=None   #Records the best state found so far
        self.improv_hist=[] #Records number of steps, reward, and accuracy when a new best state is found
        self.current_reward=None  #Current reward for guess mode

        #Initialize action and observation spaces
        self.action_space = gym.spaces.Box(
            np.float32(array([-1. for i in range(self.state_dim)])), np.float32(array([1. for i in range(self.state_dim)]))
            )
        self.observation_space = gym.spaces.Box(
            np.float32(array([-inf for i in range(self.obs_dim)])),np.float32(array([inf for i in range(self.obs_dim)]))
            )
        
    def step(self, action):
        self.step_counter+=1 #Increase step counter 
        acc=self.accuracy_fn(self.state)
        #In guess mode the episode is terminated if an improvement to the current state is found.
        if self.guess_mode:  
            current_reward=-norm(self.error_fn(self.state))*self.scale
        #In freeze_mode freeze the parameters correspoding to 0 in freeze_mask during training    
        if self.freeze_mode:
            if acc<self.freeze_accuracy and not self.guess_mode:
                self.freeze_mode=False
            action=action*self.freeze_mask
        self.state=self.state+action*self.search_size #scale the action by the search_size
        #Project the state to the specified range
        self.state = np.maximum(self.state_min, self.state) 
        self.state = np.minimum(self.state_max, self.state)
        obs=self.error_fn(self.state)
        reward = -norm(self.error_fn(self.state))*self.scale #compute and scale the reward
        #initialize the best reward
        if self.best_reward is None:
            self.best_reward=reward
            self.best_state=self.state

        if self.guess_mode:

            if self.step_counter>self.guess_mode_steps: #end guess_mode
                self.guess_mode=False
                done=True
            #check for update for the best reward
            if reward > self.best_reward:
                self.best_reward=reward
                self.best_state=self.state
                self.improv_hist.append((self.step_counter, reward, self.state, acc))
                self.not_improving_steps=0

            if reward>current_reward:
                done=True
            else:
                self.not_improving_steps+=1
                done=False

        else:
            #check for update for the best reward
            if reward > self.best_reward:
                self.best_reward=reward
                self.best_state=self.state
                self.not_improving_steps=0
                self.improv_hist.append((self.step_counter, reward, self.state,acc))
                done = True
            else:
                self.not_improving_steps+=1
                done = False
            #Terminate the episode if the reward gets too small. This should limit instabilities and divergencies
            if reward<self.reward_cap*self.best_reward: 
                    done=True
        #Check if the agent is trapped, i.e. not improving since a long time. 
        if self.not_improving_steps>self.trapped_steps:
                self.not_improving_steps=0
                done=True

        info = {}
        return obs, reward, done, info

    def reset(self):
        '''Resets the environment'''
        if self.guess_mode:
            self.state=self.initial_state
        else:
            self.state=self.best_state
        return self.error_fn(self.state)



