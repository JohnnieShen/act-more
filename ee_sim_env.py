import numpy as np
import collections
import os
import time 
import matplotlib.pyplot as plt
from constants import DT, XML_DIR, START_ARM_POSE, FR5_START_ARM_POSE, FR5_EMPTY_START_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_CLOSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

from utils import sample_box_pose, sample_insertion_pose, sample_place_box_pose
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base

import IPython
e = IPython.embed

def setup_and_print_env(task_name):
    env = make_ee_sim_env(task_name)

    # ts = env.reset()

    # ax = plt.subplot()
    # plt_img = ax.imshow(ts.observation['images']['top'])
    # plt.ion()

    physics = env.physics
    physics.named.data.qpos[:16] = FR5_START_ARM_POSE

    # for i in range(100):
    #     action = np.zeros(16)
    #     action[:16] = np.array([-0.02, -1.52, -2.63, 0.9, 1.6, 0, 0.024, -0.024, -0.02, -1.52, -2.63, 0.9, 1.6, 0, 0.024, -0.024])
    #     # action[7:7+6] = np.array([-0.02, -1.52, -2.63, 0.9, 1.6, 0])
    #     ts = env.step(action)
    #     plt_img.set_data(ts.observation['images']['top'])
    #     plt.pause(0.002)
    #     time.sleep(0.005)

    
    print("Initial joint positions (qpos):")
    print(physics.named.data.qpos[:16])
    
    print("\nInitial mocap positions (left and right end effectors):")
    print("Left end effector position:", env._physics.named.data.xpos['left_gripper_link'])
    print("Right end effector position:", env._physics.named.data.xpos['right_gripper_link'])
    
    print("\nInitial mocap orientations (left and right end effectors):")
    print("Left end effector orientation:", env._physics.named.data.xquat['left_gripper_link'])
    print("Right end effector orientation:", env._physics.named.data.xquat['right_gripper_link'])
    
    # print("\nInitial gripper control positions:")
    # print(physics.data.ctrl)


def make_ee_sim_env(task_name):
    """
    Environment for simulated robot bi-manual manipulation, with end-effector control.
    Action space:      [left_arm_pose (7),             # position and quaternion for end effector
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_pose (7),            # position and quaternion for end effector
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
    """
    if 'sim_fr5_place_cube' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_fr5_ee_place_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = FR5PickAndPlaceEETask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_transfer_cube' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_ee_transfer_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeEETask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_insertion' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_ee_insertion.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = InsertionEETask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    elif 'sim_place_cube' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_ee_place_cube.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = PickAndPlaceEETask(random=False)
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
    else:
        raise NotImplementedError
    return env

class BimanualViperXEETask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        a_len = len(action) // 2
        action_left = action[:a_len]
        action_right = action[a_len:]

        # set mocap position and quat
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # right
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])

        # set gripper
        g_left_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_left[7])
        g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[7])
        np.copyto(physics.data.ctrl, np.array([g_left_ctrl, -g_left_ctrl, g_right_ctrl, -g_right_ctrl]))

    def initialize_robots(self, physics):
        # reset joint position
        physics.reset()
        # print(physics.named.data.qpos)
        physics.named.data.qpos[:16] = FR5_START_ARM_POSE
        # print(physics.data.mocap_pos)
        # print(physics.data.mocap_quat)

        # print(physics.named.data.xpos['left_gripper_link'])
        # print(physics.named.data.xpos['right_gripper_link'])
        # print(physics.named.data.xquat['left_gripper_link'])
        # print(physics.named.data.xquat['right_gripper_link'])

        # reset mocap to align with end effector
        # to obtain these numbers:
        # (1) make an ee_sim env and reset to the same start_pose
        # (2) get env._physics.named.data.xpos['vx300s_left/gripper_link']
        #     get env._physics.named.data.xquat['vx300s_left/gripper_link']
        #     repeat the same for right side

        # however, the data generated here should be the position for the definition of the mocap bodies in the ee mjcf config file, not here (sjm)

        # np.copyto(physics.data.mocap_pos[0], [-1.57,   0.3,    0.042])
        # np.copyto(physics.data.mocap_quat[0], [-0.66446301, -0.24184481,  0.24184481,  0.66446301])
        # # right
        # np.copyto(physics.data.mocap_pos[1], [1.57,  0.9,   0.042])
        # np.copyto(physics.data.mocap_quat[1],  [-0.66446301, -0.24184481, -0.24184481, -0.66446301])

        np.copyto(physics.data.mocap_pos[0], [-0.28, 0.3, 0.6])
        np.copyto(physics.data.mocap_quat[0], [0.707, 0.707, 0, 0])
        # right
        np.copyto(physics.data.mocap_pos[1], [0.28, 0.9, 0.6])
        np.copyto(physics.data.mocap_quat[1],  [0.707, -0.707, 0, 0])

        # print(physics.data.mocap_pos)
        # print(physics.data.mocap_quat)

        # reset gripper control
        close_gripper_control = np.array([
            PUPPET_GRIPPER_POSITION_CLOSE,
            -PUPPET_GRIPPER_POSITION_CLOSE,
            PUPPET_GRIPPER_POSITION_CLOSE,
            -PUPPET_GRIPPER_POSITION_CLOSE,
        ])
        np.copyto(physics.data.ctrl, close_gripper_control)
        # print(physics.named.data.qpos)
        


    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        # note: it is important to do .copy()
        # print(physics.named.data.qpos)
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        # obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        # obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
        # used in scripted policy to obtain starting pose
        obs['mocap_pose_left'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()
        obs['mocap_pose_right'] = np.concatenate([physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()

        # used when replaying joint trajectory
        obs['gripper_ctrl'] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        raise NotImplementedError


class TransferCubeEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        cube_pose = sample_box_pose()
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_left_gripper: # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table: # successful transfer
            reward = 4
        return reward


class InsertionEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize peg and socket position
        peg_pose, socket_pose = sample_insertion_pose()
        id2index = lambda j_id: 16 + (j_id - 16) * 7 # first 16 is robot qpos, 7 is pose dim # hacky

        peg_start_id = physics.model.name2id('red_peg_joint', 'joint')
        peg_start_idx = id2index(peg_start_id)
        np.copyto(physics.data.qpos[peg_start_idx : peg_start_idx + 7], peg_pose)
        # print(f"randomized cube position to {cube_position}")

        socket_start_id = physics.model.name2id('blue_socket_joint', 'joint')
        socket_start_idx = id2index(socket_start_id)
        np.copyto(physics.data.qpos[socket_start_idx : socket_start_idx + 7], socket_pose)
        # print(f"randomized cube position to {cube_position}")

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper = ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = ("socket-1", "table") in all_contact_pairs or \
                             ("socket-2", "table") in all_contact_pairs or \
                             ("socket-3", "table") in all_contact_pairs or \
                             ("socket-4", "table") in all_contact_pairs
        peg_touch_socket = ("red_peg", "socket-1") in all_contact_pairs or \
                           ("red_peg", "socket-2") in all_contact_pairs or \
                           ("red_peg", "socket-3") in all_contact_pairs or \
                           ("red_peg", "socket-4") in all_contact_pairs
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table): # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table): # peg and socket touching
            reward = 3
        if pin_touched: # successful insertion
            reward = 4
        return reward

class PickAndPlaceEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4
        self.previous_qvel = None

    def initialize_episode(self, physics):
        self.initialize_robots(physics)
        box_pose = sample_place_box_pose()
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx: box_start_idx + 7], box_pose)
        self.place_xyz = np.array([0.2, 0.5, 0.02])
        super().initialize_episode(physics)
        self.previous_qvel = np.zeros_like(physics.data.qvel)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs
        touch_side_left = ("bin_left_wall","vx300s_right/10_right_gripper_finger") in all_contact_pairs or \
                        ("bin_left_wall","vx300s_right/10_left_gripper_finger") in all_contact_pairs
        touch_side_right = ("bin_right_wall","vx300s_right/10_right_gripper_finger") in all_contact_pairs or \
                        ("bin_right_wall","vx300s_right/10_left_gripper_finger") in all_contact_pairs
        touch_side_front = ("bin_front_wall","vx300s_right/10_right_gripper_finger") in all_contact_pairs or \
                        ("bin_front_wall","vx300s_right/10_left_gripper_finger") in all_contact_pairs
        touch_side_back = ("bin_back_wall","vx300s_right/10_right_gripper_finger") in all_contact_pairs or \
                        ("bin_back_wall","vx300s_right/10_left_gripper_finger") in all_contact_pairs
        touch_side = touch_side_left or touch_side_right or touch_side_front or touch_side_back

        box_touched_side = ("bin_left_wall","red_box") in all_contact_pairs or ("bin_right_wall","red_box") in all_contact_pairs or \
                        ("bin_front_wall","red_box") in all_contact_pairs or ("bin_back_wall","red_box") in all_contact_pairs
        
        box_xyz = physics.named.data.geom_xpos['red_box'][:3]
        placed_correctly = np.linalg.norm(box_xyz - self.place_xyz) < 0.02

        reward = 0
        if touch_right_gripper:
            # print("1")
            reward = 1  # Grasped the box with the right gripper
        if touch_right_gripper and (not touch_table):
            # print(f"2 {np.linalg.norm(box_xyz - self.place_xyz)}")
            reward = 2  # Lifted the box with the right gripper
        if touch_right_gripper and placed_correctly:
            # print(f"3 {np.linalg.norm(box_xyz - self.place_xyz)}")
            reward = 3  # Box is held by the right gripper and placed correctly
        if placed_correctly and (not touch_right_gripper):
            # print(f"4 {np.linalg.norm(box_xyz - self.place_xyz)}")
            reward = 4  # Box is placed correctly and not held by the gripper
        
        # if touch_side:
        #     print("gripper touched side")
        #     reward -= 2
        # if box_touched_side:
        #     print("box touched side")
        #     reward -= 2

        # current_qvel = physics.data.qvel
        # velocity_change = np.linalg.norm(current_qvel - self.previous_qvel)
        # if velocity_change > 0.08:
        #     penalty = -0.1 * velocity_change
            # reward += penalty
        # print(f"velocity change: {velocity_change}")

        # self.previous_qvel = current_qvel.copy()

        return reward


class FR5PickAndPlaceEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4
        self.previous_qvel = None

    def initialize_episode(self, physics):
        self.initialize_robots(physics)
        box_pose = sample_place_box_pose()
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx: box_start_idx + 7], box_pose)
        self.place_xyz = np.array([0.2, 0.5, 0.02])
        super().initialize_episode(physics)
        self.previous_qvel = np.zeros_like(physics.data.qvel)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):

        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_box", "right_10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs
        touch_side_left = ("bin_left_wall", "right_10_right_gripper_finger") in all_contact_pairs or \
                          ("bin_left_wall", "right_10_left_gripper_finger") in all_contact_pairs
        touch_side_right = ("bin_right_wall", "right_10_right_gripper_finger") in all_contact_pairs or \
                           ("bin_right_wall", "right_10_left_gripper_finger") in all_contact_pairs
        touch_side_front = ("bin_front_wall", "right_10_right_gripper_finger") in all_contact_pairs or \
                           ("bin_front_wall", "right_10_left_gripper_finger") in all_contact_pairs
        touch_side_back = ("bin_back_wall", "right_10_right_gripper_finger") in all_contact_pairs or \
                          ("bin_back_wall", "right_10_left_gripper_finger") in all_contact_pairs
        touch_side = touch_side_left or touch_side_right or touch_side_front or touch_side_back

        box_touched_side = ("bin_left_wall", "red_box") in all_contact_pairs or (
        "bin_right_wall", "red_box") in all_contact_pairs or \
                           ("bin_front_wall", "red_box") in all_contact_pairs or (
                           "bin_back_wall", "red_box") in all_contact_pairs

        box_xyz = physics.named.data.geom_xpos['red_box'][:3]
        placed_correctly = np.linalg.norm(box_xyz - self.place_xyz) < 0.02

        reward = 0
        if touch_right_gripper:
            # print("1")
            reward = 1  # Grasped the box with the right gripper
        if touch_right_gripper and (not touch_table):
            # print(f"2 {np.linalg.norm(box_xyz - self.place_xyz)}")
            reward = 2  # Lifted the box with the right gripper
        if touch_right_gripper and placed_correctly:
            # print(f"3 {np.linalg.norm(box_xyz - self.place_xyz)}")
            reward = 3  # Box is held by the right gripper and placed correctly
        if placed_correctly and (not touch_right_gripper):
            # print(f"4 {np.linalg.norm(box_xyz - self.place_xyz)}")
            reward = 4  # Box is placed correctly and not held by the gripper

        # if touch_side:
        #     print("gripper touched side")
        #     reward -= 2
        # if box_touched_side:
        #     print("box touched side")
        #     reward -= 2

        # current_qvel = physics.data.qvel
        # velocity_change = np.linalg.norm(current_qvel - self.previous_qvel)
        # if velocity_change > 0.08:
        #     penalty = -0.1 * velocity_change
        # reward += penalty
        # print(f"velocity change: {velocity_change}")

        # self.previous_qvel = current_qvel.copy()

        return reward
