from .ppo import PPO, PPOExpert
from .airl import AIRL, AIRLExpert

ALGOS = {'airl': AIRL,
         'ppo': PPO
}

EXP_ALGOS = {'ppo': PPOExpert,
             'airl': AIRLExpert
}