import os
import time
import tqdm
import h5py
import asyncio
import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
from fairino import Robot
from fairino_utils import normalize_angles, WRIST_ROTATE_CONSTANT
import argparse
from constants import DT, START_ARM_POSE, SIM_TASK_CONFIGS, FPS, FR5_START_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_CLOSE, PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
from sim_env import make_sim_env, BOX_POSE
from ee_sim_env import make_ee_sim_env
from sshkeyboard import listen_keyboard

gripper_flag = True
recording_flag = False
wrist_rotate_radian = 0

def press(key):
    global gripper_flag
    global recording_flag
    global wrist_rotate_radian
    if key == "space":
        recording_flag = True
        if gripper_flag:
            print("gripper close")
        else:
            print("gripper open")
        gripper_flag = not gripper_flag
    elif key == "a":
        if wrist_rotate_radian + WRIST_ROTATE_CONSTANT < 3.14 and wrist_rotate_radian - WRIST_ROTATE_CONSTANT> -3.14:
            wrist_rotate_radian -= WRIST_ROTATE_CONSTANT
            print(f"rotate wrist by -{WRIST_ROTATE_CONSTANT} radian")
        else:
            print("wrist rotation limit reached")
    elif key == "d":
        if wrist_rotate_radian + WRIST_ROTATE_CONSTANT < 3.14 and wrist_rotate_radian - WRIST_ROTATE_CONSTANT> -3.14:
            wrist_rotate_radian += WRIST_ROTATE_CONSTANT
            print(f"rotate wrist by {WRIST_ROTATE_CONSTANT} radian")
        else:
            print("wrist rotation limit reached")



async def listen_for_keys():
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        await loop.run_in_executor(pool, listen_keyboard, press)

async def main(args):
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

    if ((not degree) and (not radian)) or (degree and radian):
        print("Please specify either degree or radian")
        exit()

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path):
        print(f'Dataset already exists at \n{dataset_path}')
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
    BOX_POSE[0] = subtask_info
    ts = env.reset()

    # physics = env.physics
    # print(physics.named.data.qpos[:16])
    # physics.named.data.qpos[:16] = FR5_START_ARM_POSE

    render_cam_name = 'top'
    ax = plt.subplot()
    plt_img = ax.imshow(ts.observation['images'][render_cam_name])
    plt.ion()
    
    # fig, axes = plt.subplots(1, len(camera_names))
    # plt.ion()

    # if len(camera_names) == 1:
    #     axes = [axes]
    # plt_imgs = []
    # render_cams = ['top', 'left_wrist', 'right_wrist']
    
    # for ax, cam_name in zip(axes, render_cams):
    #     img = ax.imshow(ts.observation['images'][cam_name])
    #     plt_imgs.append(img)
    #     ax.set_title(cam_name)

    ret,state = robot.IsInDragTeach()
    if (state == 0):
        ret = robot.DragTeachSwitch(1)
        print("Drag teach mode switched on")
    else: 
        print("Drag teach mode already on!")
    print("Press space to start recording")

    while not recording_flag:
        await asyncio.sleep(0.1)

    # for t in tqdm(range(max_timesteps), desc="Recording episode"):
    for t in range(max_timesteps):
        ret,state = robot.IsInDragTeach()
        if (state == 0):
            ret = robot.DragTeachSwitch(1)
        t0 = time.time()
        if degree:
            ret = robot.GetActualJointPosDegree()
            print((f'Received joint degree positions: {ret} at time {t}'))
            ret = normalize_angles(list(ret))
        elif radian:
            ret = robot.GetActualJointPosRadian()
            if (type(ret) == int):
                print(f"Error fetching data, code: {ret}")
                exit()
            ret = ret[1]
            t1 = time.time()
            # print("time taken to get joint pos: ", t1-t0)
        action = np.zeros(14)
        # print((f'Received joint positions: {ret} at time {t}'))
        action[:6] = np.array([ret[0], ret[1], ret[2], ret[3], ret[4], wrist_rotate_radian])
        action[7:7+6] = np.array([-0.02, -1.52, -2.63, 0.9, 1.6, 0])
        if gripper_flag:
            action[6] = PUPPET_GRIPPER_JOINT_OPEN
        else:
            action[6] = PUPPET_GRIPPER_JOINT_CLOSE
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

        # for img, cam_name in zip(plt_imgs, render_cams):
        #     img.set_data(ts.observation['images'][cam_name])

        pic = ts.observation['images'][render_cam_name]

        t2 = time.time()
        # print("time taken to render: ", t2-t1)
        plt_img.set_data(pic)
        t3 = time.time()
        # print("time taken to display: ", t3-t2)
        plt.pause(0.002)
        await asyncio.sleep(0.005)

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
            _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3))
        # compression='gzip',compression_opts=2,)
        # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
        qpos = obs.create_dataset('qpos', (max_timesteps, 14))
        qvel = obs.create_dataset('qvel', (max_timesteps, 14))
        action = root.create_dataset('action', (max_timesteps, 14))

        for name, array in data_dict.items():
            root[name][...] = array

        print(f"Saved dataset to {dataset_path}.hdf5")
    
    ret = robot.DragTeachSwitch(0)
    print("Drag teach mode off")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--dataset_name', action='store', type=str, help='dataset_name', required=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--degree', action='store_true')
    parser.add_argument('--radian', action='store_true')
    args = vars(parser.parse_args())

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(main(args), listen_for_keys()))
    loop.close()
