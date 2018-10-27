import numpy as np
import gym
from random import randint

class Ternary(gym.Space):
    """{-1, 0, 1}^n

       For an example implementation see Discrete at:
       https://github.com/openai/gym/blob/78c416ef7bc829ce55b404b6604641ba0cf47d10/gym/spaces/discrete.py
    """
    def __init__(self, n):
        self.n = n
        gym.Space.__init__(self, (), np.int64)

    def sample(self):
        return [randint(-1,1) for _ in range(self.n)]

    def contains(self, x):
        if isinstance(x, int):
            as_int = x
        elif isinstance(x, (np.generic, np.ndarray)) and (x.dtype.kind in np.typecodes['AllInteger'] and x.shape == ()):
            as_int = int(x)
        else:
            return False

        return as_int >= -1 and as_int <= 1

    def __repr__(self):
        return "Ternary({})".format(self.n)

    def __eq__(self, other):
        return self.n == other.n
    
