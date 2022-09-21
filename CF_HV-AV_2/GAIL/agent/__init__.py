from .ppo import PPO, PPOExpert
from .gail import GAIL, GAILExpert

ALGOS = {'gail': GAIL,
         'ppo': PPO
}

EXP_ALGOS = {'ppo': PPOExpert,
             'gail': GAILExpert
}