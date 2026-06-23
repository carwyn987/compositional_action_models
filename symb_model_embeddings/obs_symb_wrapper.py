import gymnasium as gym
from gymnasium.spaces import Box, Dict
import numpy as np

class EnvWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, symb_embedding: np.ndarray):
        super().__init__(env)
        self.observation_space = Dict({
            "observation": Box(
                low=-np.inf,
                high=np.inf,
                shape=(25,),
                dtype=np.float64,
            ),
            "achieved_goal": Box(
                low=-np.inf,
                high=np.inf,
                shape=(3,),
                dtype=np.float64,
            ),
            "desired_goal": Box(
                low=-np.inf,
                high=np.inf,
                shape=(3,),
                dtype=np.float64,
            ),
            "symb_embedding": Box(
                low=-np.inf,
                high=np.inf,
                shape=symb_embedding.flatten().shape,
                dtype=np.float64,
            )
        })
        self.symb_embedding = symb_embedding
        
    def observation(self, obs):
        return obs | {"symb_embedding": self.symb_embedding}
