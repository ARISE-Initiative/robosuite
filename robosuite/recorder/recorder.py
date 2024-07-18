import h5py
import time
import numpy as np
import os


cam_height = 256
cam_width = 256
episode_len = 800

class Recorder:
    def __init__(self, cameras, task) -> None:
        self.data_dir = "data"
        self.data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        self.cameras = cameras
        self.task = task

        for cam_name in cameras:
            self.data_dict[f'/observations/images/{cam_name}'] = []
        

    def record(self, obs, action) -> None:
        qpos = np.arctan2(obs['robot0_joint_pos_sin'], obs['robot0_joint_pos_cos'])
        self.data_dict['/observations/qpos'].append(np.concatenate((qpos, obs['grasp'])))
        self.data_dict['/observations/qvel'].append(np.concatenate((obs['robot0_joint_vel'], obs['grasp'])))
        self.data_dict['/action'].append(action)
        for cam_name in self.cameras:
            self.data_dict[f'/observations/images/{cam_name}'].append(obs[cam_name + "_image"])

    def save(self) -> None:
        max_timesteps = len(self.data_dict['/observations/qpos'])
        if max_timesteps < 10:
            print('Not enough steps to save episode')
            return
        if max_timesteps > episode_len:
            print('recording longer than expected, skipping save')
            return

        # padding to episode_len        
        pad_len = episode_len - max_timesteps
        self.data_dict['/observations/qpos'] = np.pad(self.data_dict['/observations/qpos'], ((0, pad_len), (0, 0)), mode='constant')
        self.data_dict['/observations/qvel'] = np.pad(self.data_dict['/observations/qvel'], ((0, pad_len), (0, 0)), mode='constant')
        self.data_dict['/action'] = np.pad(self.data_dict['/action'], ((0, pad_len), (0, 0)), mode='constant')
        for cam_name in self.cameras:
            self.data_dict[f'/observations/images/{cam_name}'] = np.pad(self.data_dict[f'/observations/images/{cam_name}'], ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='constant')

        # create data dir if it doesn't exist
        self.data_dir = os.path.join(self.data_dir, self.task)
        if not os.path.exists(self.data_dir): os.makedirs(self.data_dir)
        # count number of files in the directory
        idx = len([name for name in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, name))])
        dataset_path = os.path.join(self.data_dir, f'episode_{idx}')
        print(f"Saving episode to {dataset_path}")
        # save the data
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in self.cameras:
                _ = image.create_dataset(cam_name, (episode_len, cam_height, cam_width, 3), dtype='uint8',
                                        chunks=(1, cam_height, cam_width, 3), )
            qpos = obs.create_dataset('qpos', (episode_len, 8))
            qvel = obs.create_dataset('qvel', (episode_len, 8))
            # image = obs.create_dataset("image", (episode_len, 240, 320, 3), dtype='uint8', chunks=(1, 240, 320, 3))
            action = root.create_dataset('action', (episode_len, 7))
            
            for name, array in self.data_dict.items():
                root[name][...] = array