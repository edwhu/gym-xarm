import numpy as np

from gym_xarm.tasks import Base


class Lift(Base):
    metadata = {
        **Base.metadata,
        "action_space": "xyzw",
        "episode_length": 50,
        "description": "Lift a cube above a height threshold",
    }

    def __init__(self, **kwargs):
        self._z_threshold = 0.2
        super().__init__("lift", **kwargs)

    def _initialize_simulation(self):
        """Initialize MuJoCo simulation data structures mjModel and mjData."""
        self.model = self._mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = self._mujoco.MjData(self.model)

        # make table rainbow shaped.
        l = len(self.model.tex_type)
        for i in range(l):
            if self.model.tex_type[i] == 0:
                height = self.model.tex_height[i]
                width = self.model.tex_width[i]
                s = self.model.tex_adr[i]
                for x in range(height):
                    for y in range(width):
                        cur_s = s + (x * width + y) * 3
                        self.model.tex_data[cur_s:cur_s + 3] = [int(x / height * 255), int(y / width * 255), 128]
        self.model.mat_texrepeat[:, :] = 1

        # make robot transparent
        robot_geom_names = ['bb', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j8', 'j9', 'right_outer_knuckle', 'right_inner_knuckle', 'left_outer_knuckle', 'left_inner_knuckle']
        for g in robot_geom_names:
            geom_id = self.model.geom(g).id
            self.model.geom_rgba[geom_id][3] = 0.0

        self._model_names = self._utils.MujocoModelNames(self.model)

        self.model.vis.global_.offwidth = self.observation_width
        self.model.vis.global_.offheight = self.observation_height

        self._env_setup(initial_qpos=self.initial_qpos)
        self.initial_time = self.data.time
        self.initial_qpos = np.copy(self.data.qpos)
        self.initial_qvel = np.copy(self.data.qvel)

    @property
    def z_target(self):
        return self._init_z + self._z_threshold - self.center_of_table[2]

    def is_success(self):
        # print(self.obj[2], self.z_target)
        return self.obj[2] >= (self.z_target)

    def get_reward(self):
        # reach_dist = np.linalg.norm(self.obj - self.eef)
        # reach_dist_xy = np.linalg.norm(self.obj[:-1] - self.eef[:-1])
        # pick_completed = self.obj[2] >= (self.z_target - 0.01)
        # obj_dropped = (self.obj[2] < (self._init_z + 0.005)) and (reach_dist > 0.02)

        # # Reach
        # if reach_dist < 0.05:
        #     reach_reward = -reach_dist + max(self._action[-1], 0) / 50
        # elif reach_dist_xy < 0.05:
        #     reach_reward = -reach_dist
        # else:
        #     z_bonus = np.linalg.norm(np.linalg.norm(self.obj[-1] - self.eef[-1]))
        #     reach_reward = -reach_dist - 2 * z_bonus

        # # Pick
        # if pick_completed and not obj_dropped:
        #     pick_reward = self.z_target
        # elif (reach_dist < 0.1) and (self.obj[2] > (self._init_z - self.center_of_table[2] + 0.005)):
        #     pick_reward = min(self.z_target, self.obj[2])
        # else:
        #     pick_reward = 0
        # # print('picked', pick_completed, 'dropped', obj_dropped, 'reach', reach_reward, 'pick', pick_reward)
        # return reach_reward / 100 + pick_reward
        return float(self.is_success())

    def _get_obs(self):
        return np.concatenate(
            [
                self.eef,
                # self.eef_velp,
                self.obj,
                # self.obj_rot,
                # self.obj_velp,
                # self.obj_velr,
                # self.eef - self.obj,
                # np.array(
                #     [
                #         np.linalg.norm(self.eef - self.obj),
                #         np.linalg.norm(self.eef[:-1] - self.obj[:-1]),
                #         self.z_target,
                #         self.z_target - self.obj[-1],
                #         self.z_target - self.eef[-1],
                #     ]
                # ),
                self.gripper_angle,
            ],
            axis=0,
        )

    def _sample_goal(self):
        # Gripper
        gripper_pos = np.array([1.280, 0.295, 0.735]) + self.np_random.uniform(-0.05, 0.05, size=3)
        super()._set_gripper(gripper_pos, self.gripper_rotation)

        # Object
        object_pos = self.center_of_table - np.array([0.15, 0.10, 0.07])
        object_pos[0] += self.np_random.uniform(-0.05, 0.05)
        object_pos[1] += self.np_random.uniform(-0.05, 0.05)
        object_qpos = self._utils.get_joint_qpos(self.model, self.data, "object_joint0")
        object_qpos[:3] = object_pos
        self._utils.set_joint_qpos(self.model, self.data, "object_joint0", object_qpos)
        self._init_z = object_pos[2]

        # Goal
        return object_pos + np.array([0, 0, self._z_threshold])

    def reset(
        self,
        seed=None,
        options: dict | None = None,
    ):
        self._action = np.zeros(4)
        return super().reset(seed=seed, options=options)

    def step(self, action):
        self._action = action.copy()
        return super().step(action)
   
    def _initialize_rendering(self):
        # obs_renderer_kwargs = {"camera_name": "top_camera", "width": self.observation_width, "height": self.observation_height}
        obs_renderer_kwargs = {"camera_name": "camera0", "width": self.observation_width, "height": self.observation_height}
        vis_renderer_kwargs = {"camera_name": "camera0", "width": self.visualization_width, "height": self.visualization_width}
        self.observation_renderer = self._initialize_renderer(renderer_type="observation", renderer_kwargs=obs_renderer_kwargs)
        self.visualization_renderer = self._initialize_renderer(renderer_type="visualization", renderer_kwargs=vis_renderer_kwargs)
