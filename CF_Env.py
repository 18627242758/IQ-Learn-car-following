from gym import spaces
import numpy as np

class CF_Env(object):
    """
    Car-following Imitation Learning Environment
    """
    def __init__(self):
        self.spacing_LowerBound = 0
        self.spacing_HigherBound = 70
        self.SvSpd_LowerBound = 3
        self.SvSpd_HigherBound = 35
        self.relSpd_LowerBound = -2
        self.relSpd_HigherBound = 2
        # the observation is spacing, following vehicle speed, and relative speed (following vehicle speed - leading vehicle speed)
        self.observation_space = spaces.Box(
            low=np.array([self.spacing_LowerBound,self.SvSpd_LowerBound,self.relSpd_LowerBound]),
            high=np.array([self.spacing_HigherBound,self.SvSpd_HigherBound,self.relSpd_HigherBound]),
            shape=(3,), dtype=np.float32
        )

        self.action_LowerBound = -3
        self.action_HigherBound = 3
        # the action is the acceleration of the following vehicle
        self.action_space = spaces.Box(
            low=self.action_LowerBound, high=self.action_HigherBound, shape=(1,), dtype=np.float32
        )

    def reset(self, data):
        """
        Reset the environment at a given time step
        """
        # initial timestep
        self.timeStep = 1
        # check if there is a crash
        self.isCollision = 0
        # car-following event length
        self.TimeLen = data.shape[0]

        # spacing
        # initialize with initial spacing
        self.RealSpaceData = data[:, 0]
        self.SimSpaceData = np.zeros(data[:, 0].shape)
        self.SimSpaceData[0] = data[0, 0]

        # following vehicle speed
        # initialize with initial following vehicle speed
        self.RealSpeedData = data[:, 1]
        self.SimSpeedData = np.zeros(data[:, 1].shape)
        self.SimSpeedData[0] = data[0, 1]

        # leading vehicle speed
        self.LVSpdData = data[:, 3]

        # initial state
        self.state = data[:1, :3].reshape(1, 3)[0, :]
        self.CurrentState = self.state[-3:]

        return self.state

    def get_reward(self):
        """
        Return the rewards at a given state and action
        """
        return 0

    def _is_terminal(self):
        """
        The episode is over if the ego vehicle crashed or the time is out.
        """
        return self.isCollision or self.timeStep >= self.TimeLen

    def step(self, action):
        """
        Perform an MDP step
        """
        # update timestep
        self.timeStep += 1

        # update leading vehicle speed
        LVSpd = self.LVSpdData[self.timeStep-1]
        # update following vehicle speed
        SvSpd = self.CurrentState[1] + action*0.1
        # check if the vehicle is stalled
        if SvSpd <= 0:
            SvSpd = 1

        # update relative speed
        # following vehicle speed - leading vehicle speed
        relSpd = SvSpd - LVSpd

        # update spacing
        # assuming they are driving at constant acceleration in each time interval
        # calculate the moving distance of following vehicle
        SvDist = (SvSpd + self.CurrentState[1])*0.1/2
        # calculate the moving distance of leading vehicle
        LVDist = (LVSpd + self.LVSpdData[self.timeStep-2])*0.1/2
        space = self.CurrentState[0] - SvDist + LVDist

        # update current state
        self.CurrentState = [space, SvSpd, relSpd]

        # check if there is a crash
        if space < 0:
            self.isCollision = 1
        # judge the end
        next_state = self.CurrentState
        reward = self.get_reward()
        terminal = self._is_terminal()

        if terminal:
            return next_state, reward, True, {} # next_state, reward, done, info
        else:
            return next_state, reward, False, {} # next_state, reward, done, info
