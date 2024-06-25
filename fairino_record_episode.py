from fairino import Robot
import os
import time
import h5py
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from fairino_utils import normalize_angles
import argparse
# from frhal_msgs.msg import FRState
from constants import DT, START_ARM_POSE, SIM_TASK_CONFIGS, FPS, FR5_START_ARM_POSE
from sim_env import make_sim_env, BOX_POSE
from ee_sim_env import make_ee_sim_env

def main(args):
    robot = Robot.RPC('192.168.31.202')

    task_name = args['task_name']
    dataset_name = args['dataset_name']
    debug = args['debug']
    degree = args['degree']
    radian = args['radian']
    task_config = SIM_TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    max_timesteps = task_config['episode_len']
    camera_names = task_config['camera_names']

    if ((not degree) and (not radian)) or (degree and radian) :
        print("Please specify either degree or radian")
        exit()

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path):
        print(f'Dataset already exist at \n{dataset_path}')
        exit()

    if debug:
        print("start listening for data")
        while True:
            if degree:
                ret = robot.GetActualJointPosDegree()
                ret = normalize_angles(list(ret))
            elif radian:
                ret = robot.GetActualJointPosRadian()
            print("joints: ", ret)

    timesteps = []
    actions = []

    print("making ee_sim_env")
    ee_env = make_ee_sim_env(task_name)
    ts = ee_env.reset()
    episode = [ts]
    subtask_info = episode[0].observation['env_state'].copy()
    del ee_env, episode
    print("making sim_env")
    env = make_sim_env(task_name)
    # render_cam_name = 'top'
    BOX_POSE[0] = subtask_info
    ts = env.reset()

    # physics = env.physics
    # print(physics.named.data.qpos[:16])
    # physics.named.data.qpos[:16] = FR5_START_ARM_POSE

    render_cam_name = 'top'
    # fig, axes = plt.subplots(1, len(camera_names))
    # plt.ion()

    # if len(camera_names) == 1:
    #     axes = [axes]

    # plt_imgs = []
    # render_cams = ['top']
    ax = plt.subplot()
    plt_img = ax.imshow(ts.observation['images'][render_cam_name])
    plt.ion()
    # for ax, cam_name in zip(axes, render_cams):
    #     img = ax.imshow(ts.observation['images'][cam_name])
    #     plt_imgs.append(img)
    #     ax.set_title(cam_name)

    for t in range(max_timesteps):
        t0 = time.time()
        if degree:
            ret = robot.GetActualJointPosDegree()
            print((f'Received joint degree positions: {ret} at time {t}'))
            ret = normalize_angles(list(ret))
        elif radian:
            ret = robot.GetActualJointPosRadian()
            if (type(ret) == int):
                print(f"error: {ret}")
                exit()
            ret = ret[1]
            t1 = time.time()
            # print("time taken to get joint pos: ", t1-t0)
        action = np.zeros(14)
        # print((f'Received joint positions: {ret} at time {t}'))
        action[:6] = np.array([ret[0], ret[1], ret[2], ret[3], ret[4], ret[5]])
        action[7:7+6] = np.array([-0.02, -1.52, -2.63, 0.9, 1.6, 0])
        print((f'Sending joint positions: {action} at time {t}'))
        ts = env.step(action)
        timesteps.append(ts)
        actions.append(action)
        # print("\nInitial mocap positions (left and right end effectors):")
        # print("Left end effector position:", env._physics.named.data.xpos['left_gripper_link'])
        # print("Right end effector position:", env._physics.named.data.xpos['right_gripper_link'])
        
        # print("\nInitial mocap orientations (left and right end effectors):")
        # print("Left end effector orientation:", env._physics.named.data.xquat['left_gripper_link'])
        # print("Right end effector orientation:", env._physics.named.data.xquat['right_gripper_link'])
        # # for img, cam_name in zip(plt_imgs, render_cams):
        #     img.set_data(ts.observation['images'][cam_name])
        pic = ts.observation['images'][render_cam_name]
        t2 = time.time()
        # print("time taken to render: ", t2-t1)
        plt_img.set_data(pic)
        t3 = time.time()
        # print("time taken to display: ", t3-t2)
        print()
        plt.pause(0.002)
        time.sleep(0.005)

    plt.close()

    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        # '/observations/effort': [],
        '/action': []
    }
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []
    
    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)
        data_dict['/observations/qpos'].append(ts.observation['qpos'])
        data_dict['/observations/qvel'].append(ts.observation['qvel'])
        # data_dict['/observations/effort'].append(ts.observation['effort'])
        data_dict['/action'].append(action)
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

    # dataset_path = os.path.join(dataset_dir, dataset_name)
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = True
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in camera_names:
            _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                    chunks=(1, 480, 640, 3), )
        # compression='gzip',compression_opts=2,)
        # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
        qpos = obs.create_dataset('qpos', (max_timesteps, 14))
        qvel = obs.create_dataset('qvel', (max_timesteps, 14))
        action = root.create_dataset('action', (max_timesteps, 14))

        for name, array in data_dict.items():
            root[name][...] = array

        print(f"Saved dataset to {dataset_path}.hdf5")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    # parser.add_argument('--dataset_dir', action='store', type=str, help='dataset_dir', required=True)
    parser.add_argument('--dataset_name', action='store', type=str, help='dataset_name', required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--degree', action='store_true')
    parser.add_argument('--radian', action='store_true')
    main(vars(parser.parse_args()))