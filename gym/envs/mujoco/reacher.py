import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

pi = np.pi


class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    ee_name = 'fingertip'
    rew_ctrl_scale = 1.0

    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

    def _step(self, a):
        vec = \
            self.get_body_com(self.ee_name) - \
            self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = -self.rew_ctrl_scale * np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        cam = self.viewer.cam
        cam.trackbodyid = 0
        cam.elevation = -90
        cam.distance = 0.8

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
        self.set_geom_pos('marker', [0.5, 0.5, 0.01])
        self.set_geom_pos('obs', [0.25, 0.25, 0.01])
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
    rew_ctrl_scale = 0.1

    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher_push.xml', 2)

    def reset_model(self):
        qpos = self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nq
        ) + self.init_qpos

        while True:
            self.goal = np.random.uniform(-0.2, 0.2, size=(2,))
            if np.linalg.norm(self.goal) < 2:
                break

        qpos[2:4] = self.goal
        r = np.random.uniform(0.05, 0.15)
        th = np.random.uniform(pi / 6, pi / 2)
        qpos[4:6] = [r * np.cos(th), r * np.sin(th)]

        qvel = self.init_qvel + \
            self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[2:] = 0

        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        super(ReacherPushEnv, self).viewer_setup()
        self.viewer.cam.distance = 0.7
