import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    ee_name = 'fingertip'

    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

    def _step(self, a):
        vec = \
            self.get_body_com(self.ee_name) - \
            self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 2:
                break
        qpos[2:4] = self.goal
        qvel = self.init_qvel + \
            self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[2:4],
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target"),
            self.get_geom_pos("obs"),
        ])


class ReacherBoundedEnv(ReacherEnv):

    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher_bounded.xml', 2)


class ReacherPushEnv(ReacherEnv):

    ee_name = 'obj'

    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher_push.xml', 2)

    def reset_model(self):
        super(ReacherPushEnv, self).reset_model()

        qpos = self.data.qpos.ravel().copy()
        qvel = self.data.qvel.ravel().copy()
        r = np.random.uniform(0.05, 0.15)
        th = np.sign(np.random.random() - 0.5) \
            * np.random.uniform(np.pi / 6, np.pi / 2)
        qpos[4:6] = [r * np.cos(th), r * np.sin(th)]
        qvel[2:] = 0

        self.set_state(qpos, qvel)
        return self._get_obs()
