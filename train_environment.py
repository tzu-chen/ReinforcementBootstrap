import gym
import numpy as np
from numpy import array, inf


class TrainingEnv(gym.Env):
    """
    Creates a general training environment that can be used with different CFTs.
    The input to the Network are the states, not the errors for the crossing equations.
    """     
    def __init__(self, error_fn, accuracy_fn, state_dim, initial_state, 
                 min_state_bound=None, max_state_bound=None,
                 guess_mode_steps=0, trapped_steps=100, reward_cap=100, reward_scale=1.,
                 guess_size=6.5, freeze_mask=None, freeze_accuracy=0.01):
        """
        Args:
        error_fn:        function for the crossing equations to be used for penalty.
                         Given a state as input should return ||E||.
        accuracy_fn:     function used to compute accuracy. Given a state should return ||E||/E_{abs}
        state_dim:       integer. dimension of the array of variables to be searched.
        initial state:   array of dimension state_dim. Initial values for the variables to be searched.
        min_state_bound: float or array of dimension state_dim. Sets the minimum value for the search of variables. 
                         If a float, the bound is common for all variables.
        max_state_bound: float or array of dimension state_dim. Sets the maximum value for the search of variables. 
                         If a float, the bound is common for all variables.
        gues_mode_steps: integer. Number of steps to be taken in guess mode.
        trapped_steps:   integer. Number of steps without improvement on the best state after which the environment is reset. 
        reward_cap:      float. Resets the environment when the agent finds a state with reward<reward_cap*best_reward. Should increase stability.
        reward_scale:    float. Reward is computed as -reward_scale*||E||
        guess_size:      float or array of dimension state_dim. Sets the search window at each step for the variables.
                     If a float, the guess size is the same for all variables.
        freeze_mask:     array of dimension state_dim containing only 0 and 1. The variables corresponding to indices that are 0 are not changed
                     until the accuracy reach the level freeze_accuracy. This is needed for training in mode 2. If None no variable is frozen,
        freeze_accuracy: float. Value of accuracy at which all the variables are unfrozen if freeze_mask is not None
        """
        self.state_min=min_state_bound #Set minimum for the search window
        self.state_max=max_state_bound #Set maximum for the search window
        self.state_dim=state_dim
        self.best_reward=None  #Records best reward
        self.best_state=None   #Records best state
        self.scale=reward_scale #Scale the error
        self.guess_mode=True #Controls wheter the algorithm is working in guess mode
        self.guess_mode_steps=guess_mode_steps #Number of steps to take in guess mode
        self.step_counter=0 #Records the number of steps executed in the environment
        self.not_improving_steps=0 #Stores the number of steps since last improvement
        self.trapped_steps=trapped_steps #Maximum number of not improving steps after which the environment is reset
        self.current_reward=None  #Current reward for guess mode
        self.improv_hist=[] #Records reward when a new best state is found
        self.reward_cap=reward_cap #Reset the environment when reward>reward_cap*best_reward
        self.guess_size=guess_size #Search window for each parameter
        self.initial_state=initial_state
        self.error_fn=error_fn
        self.accuracy_fn=accuracy_fn
        #Setup freeze mode is a mask is passed
        if freeze_mask is not None:
                self.freeze_mode=True      
                self.freeze_mask=freeze_mask
                self.freeze_accuracy=freeze_accuracy
        else:
            self.freeze_mode=False    
        #Initialize action and observation spaces
        self.action_space = gym.spaces.Box(
            np.float32(array([-1. for i in range(self.state_dim)])), np.float32(array([1. for i in range(self.state_dim)]))
            )
        self.observation_space = gym.spaces.Box(
            np.float32(array([-inf for i in range(self.state_dim)])),np.float32(array([inf for i in range(self.state_dim)]))
            )
        
    def step(self, action):
        self.step_counter+=1 #Increase step counter 

        if self.guess_mode:
            current_reward=-self.error_fn(self.state)*self.scale
            
        if self.freeze_mode:#In freeze_mode freeze some of the parameters during training
            acc=self.accuracy_fn(self.state)
            if acc<self.freeze_accuracy and not self.guess_mode:
                self.freeze_mode=False
            action=action*self.freeze_mask
                
        self.state=self.state+action*self.guess_size #scale the action by the guess_size
        self.state = np.maximum(self.state_min, self.state)
        self.state = np.minimum(self.state_max, self.state)
        obs=self.state
        reward = -self.error_fn(self.state)*self.scale
        
        if self.best_reward is None:
            self.best_reward=reward
            self.best_state=self.state

        if self.guess_mode:

            if self.step_counter>self.guess_mode_steps: #end guess_mode
                self.guess_mode=False
                done=True
            #check update best reward
            if reward > self.best_reward:
                self.best_reward=reward
                self.best_state=self.state
                self.improv_hist.append((self.step_counter, reward, self.state))
                self.not_improving_steps=0


            if reward>current_reward:
                done=True
            else:
                self.not_improving_steps+=1
                done=False

        else:
            #check update best reward and terminate
            if reward > self.best_reward:
                self.best_reward=reward
                self.best_state=self.state
                self.not_improving_steps=0
                self.improv_hist.append((self.step_counter, reward, self.state))
                done = True
            else:
                self.not_improving_steps+=1
                done = False
                
            if reward<self.reward_cap*self.best_reward: #This should limit instabilities and divergencies
                    done=True

        if self.not_improving_steps>self.trapped_steps:
                self.not_improving_steps=0
                done=True

        info = {}
        return obs, reward, done, info

    def reset(self):
        if self.guess_mode:
            #self.state = rand(10)*6.5 #reset to a random state only in guessing mode
            self.state=self.initial_state
        else:
            self.state=self.best_state
        return self.state




