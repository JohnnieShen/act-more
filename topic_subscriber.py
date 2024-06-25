import os
import time
import h5py
from tqdm import tqdm
import numpy as np
import rclpy
import matplotlib.pyplot as plt
from rclpy.node import Node
import argparse
from frhal_msgs.msg import FRState
from constants import DT, START_ARM_POSE, SIM_TASK_CONFIGS, FPS
from sim_env import make_sim_env, BOX_POSE
import cv2
import time

class JointPositionRecorder(Node):
    def __init__(self, task_name):
        super().__init__('joint_position_recorder')
        self.subscription = self.create_subscription(
            FRState,
            'nonrt_state_data',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.timesteps = []
        self.actions = []
        self.DT = 1 / 50
        self.env = make_sim_env(task_name)
        self.render_cam_name = 'top'
        # BOX_POSE[0] = subtask_info
        ts = self.env.reset()
        ax = plt.subplot()
        self.plt_img = ax.imshow(ts.observation['images'][self.render_cam_name])
        plt.ion()

    def listener_callback(self, msg):
        t0 = time.time()
        joint_positions = {
            'j1': msg.j1_cur_pos,
            'j2': msg.j2_cur_pos,
            'j3': msg.j3_cur_pos,
            'j4': msg.j4_cur_pos,
            'j5': msg.j5_cur_pos,
            'j6': msg.j6_cur_pos,
        }
        action = np.zeros(14)
        action[:6] = np.array([msg.j1_cur_pos, msg.j2_cur_pos, msg.j3_cur_pos, msg.j4_cur_pos, msg.j5_cur_pos, msg.j6_cur_pos])
        # t1 = time.time() #
        ts = self.env.step(action)
        # t2 = time.time() #
        self.timesteps.append(ts)
        self.actions.append(action)
        self.plt_img.set_data(ts.observation['images'][self.render_cam_name])
        plt.pause(0.002)
        time.sleep(max(0, self.DT - (time.time() - t0)))
        self.get_logger().info(f'Received joint positions: {joint_positions}')

    def store_data(self, dataset_dir, dataset_name, task_name, camera_names, max_timesteps):
        if not os.path.isdir(dataset_dir):
            os.makedirs(dataset_dir)
        dataset_path = os.path.join(dataset_dir, dataset_name)
        if os.path.isfile(dataset_path) and not overwrite:
            print(f'Dataset already exist at \n{dataset_path}\nHint: set overwrite to True.')
            exit()
        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/observations/effort': [],
            '/action': [],
        }
        while self.actions:
            ts = self.timesteps.pop(0)
            action = self.actions.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/observations/effort'].append(ts.observation['effort'])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        COMPRESS = True

        if COMPRESS:
            # JPEG compression
            t0 = time.time()
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50] # tried as low as 20, seems fine
            compressed_len = []
            for cam_name in camera_names:
                image_list = data_dict[f'/observations/images/{cam_name}']
                compressed_list = []
                compressed_len.append([])
                for image in image_list:
                    result, encoded_image = cv2.imencode('.jpg', image, encode_param) # 0.02 sec # cv2.imdecode(encoded_image, 1)
                    compressed_list.append(encoded_image)
                    compressed_len[-1].append(len(encoded_image))
                data_dict[f'/observations/images/{cam_name}'] = compressed_list
            print(f'compression: {time.time() - t0:.2f}s')

            # pad so it has same length
            t0 = time.time()
            compressed_len = np.array(compressed_len)
            padded_size = compressed_len.max()
            for cam_name in camera_names:
                compressed_image_list = data_dict[f'/observations/images/{cam_name}']
                padded_compressed_image_list = []
                for compressed_image in compressed_image_list:
                    padded_compressed_image = np.zeros(padded_size, dtype='uint8')
                    image_len = len(compressed_image)
                    padded_compressed_image[:image_len] = compressed_image
                    padded_compressed_image_list.append(padded_compressed_image)
                data_dict[f'/observations/images/{cam_name}'] = padded_compressed_image_list
            print(f'padding: {time.time() - t0:.2f}s')

        t0 = time.time()
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
            root.attrs['sim'] = False
            root.attrs['compress'] = COMPRESS
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                if COMPRESS:
                    _ = image.create_dataset(cam_name, (max_timesteps, padded_size), dtype='uint8',
                                            chunks=(1, padded_size), )
                else:
                    _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                            chunks=(1, 480, 640, 3), )
            _ = obs.create_dataset('qpos', (max_timesteps, 14))
            _ = obs.create_dataset('qvel', (max_timesteps, 14))
            _ = obs.create_dataset('effort', (max_timesteps, 14))
            _ = root.create_dataset('action', (max_timesteps, 14))
            _ = root.create_dataset('base_action', (max_timesteps, 2))
            # _ = root.create_dataset('base_action_t265', (max_timesteps, 2))

            for name, array in data_dict.items():
                root[name][...] = array

            if COMPRESS:
                _ = root.create_dataset('compress_len', (len(camera_names), max_timesteps))
                root['/compress_len'][...] = compressed_len

        # print(f'Saving: {time.time() - t0:.1f} secs')

        return True
        
        
    
def main(args=None):
    task_name = args['task_name']
    task_config = SIM_TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    max_timesteps = task_config['episode_len']
    camera_names = task_config['camera_names']

    rclpy.init()
    print("rclpy initialized!")
    recorder = JointPositionRecorder(task_name)

    try:
        rclpy.spin(recorder)
    except KeyboardInterrupt:
        plt.close()
        recorder.store_data(dataset_dir,"fr5_sim_teleop", task_name, camera_names, max_timesteps)
    finally:
        recorder.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--inject_noise', action='store_true')
    parser.add_argument('--no_generate', action='store_true')
    parser.add_argument('--debug', action='store_true')
    main()
