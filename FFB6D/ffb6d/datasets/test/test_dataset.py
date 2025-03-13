#!/usr/bin/env python3
import os
import cv2
import torch
import os.path
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from common import Config
import pickle as pkl
from utils.basic_utils import Basic_Utils
import yaml
import scipy.io as scio
import scipy.misc
from glob import glob
from termcolor import colored
import normalSpeed
from models.RandLA.helper_tool import DataProcessing as DP

try:
    from neupeak.utils.webcv2 import imshow, waitKey
except ImportError:
    from cv2 import imshow, waitKey


class Dataset():
    shuffle_seed = np.random.randint(1, 10000)
    def __init__(self, dataset_name, cls_type="duck", DEBUG=False):
        self.DEBUG = DEBUG
        self.config = Config(ds_name='test', cls_type=cls_type)
        self.bs_utils = Basic_Utils(self.config)
        self.dataset_name = dataset_name
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.224])
        self.obj_dict = self.config.test_obj_dict

        self.cls_type = cls_type
        self.cls_id = self.obj_dict[cls_type]
        print("cls_id in test_dataset.py", self.cls_id)
        self.root = os.path.join(self.config.test_root, 'test_preprocessed')
        self.cls_root = os.path.join(self.root, "data/%02d/" % self.cls_id)
        self.rng = np.random
        if dataset_name == 'train':
            self.add_noise = True
            real_img_pth = os.path.join(
                self.cls_root, "train.txt"
            )
            self.real_lst = [] #self.bs_utils.read_lines(real_img_pth)

            rnd_img_ptn = os.path.join(
                self.root, 'renders/%s/*.pkl' % cls_type
            )
            self.rnd_lst = glob(rnd_img_ptn)
            print("render data length: ", len(self.rnd_lst))
            if len(self.rnd_lst) == 0:
                warning = "Warning: "
                warning += "Trainnig without rendered data will hurt model performance \n"
                warning += "Please generate rendered data from https://github.com/ethnhe/raster_triangle.\n"
                print(colored(warning, "red", attrs=['bold']))

            fuse_img_ptn = os.path.join(
                self.root, 'fuse/%s/*.pkl' % cls_type
            )
            self.fuse_lst = glob(fuse_img_ptn)
            print("fused data length: ", len(self.fuse_lst))
            if len(self.fuse_lst) == 0:
                warning = "Warning: "
                warning += "Trainnig without fused data will hurt model performance \n"
                warning += "Please generate fused data from https://github.com/ethnhe/raster_triangle.\n"
                print(colored(warning, "red", attrs=['bold']))

            # Select 70% of the rendered data as training data and rest as validation data
            np.random.seed(self.shuffle_seed)
            print(self.shuffle_seed)
            np.random.shuffle(self.rnd_lst)
            num_train_rnd = int(len(self.rnd_lst) * 0.7)
            self.rnd_lst = self.rnd_lst[:num_train_rnd]

            self.all_lst = self.real_lst + self.rnd_lst + self.fuse_lst
            self.minibatch_per_epoch = len(self.all_lst) // self.config.mini_batch_size
        else:
            self.add_noise = False

            tst_img_pth = os.path.join(
                self.root, 'renders/%s/*.pkl' % cls_type
            )
            self.tst_lst = glob(tst_img_pth)
            print("render data length: ", len(self.tst_lst))
            if len(self.tst_lst) == 0:
                warning = "Warning: "
                warning += "Trainnig without rendered data will hurt model performance \n"
                warning += "Please generate rendered data from https://github.com/ethnhe/raster_triangle.\n"
                print(colored(warning, "red", attrs=['bold']))

            # Select 70% of the rendered data as training data and rest as validation data
            np.random.seed(self.shuffle_seed)
            print(self.shuffle_seed)
            np.random.shuffle(self.tst_lst)
            num_train_rnd = int(len(self.tst_lst) * 0.7)
            self.tst_lst = self.tst_lst[num_train_rnd:]

            self.all_lst = self.tst_lst
        print("{}_dataset_size: ".format(dataset_name), len(self.all_lst))

    def real_syn_gen(self, real_ratio=0): # 0.3 in normal
        if len(self.rnd_lst+self.fuse_lst) == 0:
            real_ratio = 1.0
        if self.rng.rand() < real_ratio:  # real
            n_imgs = len(self.real_lst)
            idx = self.rng.randint(0, n_imgs)
            pth = self.real_lst[idx]
            return pth
        else:
            if len(self.fuse_lst) > 0 and len(self.rnd_lst) > 0:
                fuse_ratio = 0.4
            elif len(self.fuse_lst) == 0:
                fuse_ratio = 0.
            else:
                fuse_ratio = 1.
            if self.rng.rand() < fuse_ratio:
                idx = self.rng.randint(0, len(self.fuse_lst))
                pth = self.fuse_lst[idx]
            else:
                idx = self.rng.randint(0, len(self.rnd_lst))
                pth = self.rnd_lst[idx]
            return pth

    def real_gen(self):
        n = len(self.real_lst)
        idx = self.rng.randint(0, n)
        item = self.real_lst[idx]
        return item

    def rand_range(self, rng, lo, hi):
        return rng.rand()*(hi-lo)+lo

    def gaussian_noise(self, rng, img, sigma):
        """add gaussian noise of given sigma to image"""
        img = img + rng.randn(*img.shape) * sigma
        img = np.clip(img, 0, 255).astype('uint8')
        return img

    def linear_motion_blur(self, img, angle, length):
        """:param angle: in degree"""
        rad = np.deg2rad(angle)
        dx = np.cos(rad)
        dy = np.sin(rad)
        a = int(max(list(map(abs, (dx, dy)))) * length * 2)
        if a <= 0:
            return img
        kern = np.zeros((a, a))
        cx, cy = a // 2, a // 2
        dx, dy = list(map(int, (dx * length + cx, dy * length + cy)))
        cv2.line(kern, (cx, cy), (dx, dy), 1.0)
        s = kern.sum()
        if s == 0:
            kern[cx, cy] = 1.0
        else:
            kern /= s
        return cv2.filter2D(img, -1, kern)

    def rgb_add_noise(self, img):
        rng = self.rng
        # apply HSV augmentor
        if rng.rand() > 0:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.uint16)
            hsv_img[:, :, 1] = hsv_img[:, :, 1] * self.rand_range(rng, 1-0.25, 1+.25)
            hsv_img[:, :, 2] = hsv_img[:, :, 2] * self.rand_range(rng, 1-.15, 1+.15)
            hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 255)
            hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2], 0, 255)
            img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if rng.rand() > 0.8:  # motion blur
            r_angle = int(rng.rand() * 360)
            r_len = int(rng.rand() * 15) + 1
            img = self.linear_motion_blur(img, r_angle, r_len)

        if rng.rand() > 0.8:
            if rng.rand() > 0.2:
                img = cv2.GaussianBlur(img, (3, 3), rng.rand())
            else:
                img = cv2.GaussianBlur(img, (5, 5), rng.rand())

        return np.clip(img, 0, 255).astype(np.uint8)

    def add_real_back(self, rgb, labels, dpt, dpt_msk):
        # Generate a random real image path from the dataset
        real_item = self.real_gen()

        # Load the depth image of the real background
        with Image.open(os.path.join(self.cls_root, "depth", real_item + '.png')) as di:
            real_dpt = np.array(di)

        # Load the mask of the real background
        with Image.open(os.path.join(self.cls_root, "mask", real_item + '.png')) as li:
            bk_label = np.array(li)

        # Convert the mask to a binary format (0 for background, 1 for foreground)
        bk_label = (bk_label < 255).astype(rgb.dtype)

        # Ensure the mask is 2D (if it has extra dimensions, take the first channel)
        if len(bk_label.shape) > 2:
            bk_label = bk_label[:, :, 0]

        # Load the RGB background image and apply the mask to remove background pixels
        with Image.open(os.path.join(self.cls_root, "rgb", real_item + '.png')) as ri:
            back = np.array(ri)[:, :, :3] * bk_label[:, :, None]

        # Convert the real depth image to float and apply the mask to remove background depth values
        dpt_back = real_dpt.astype(np.float32) * bk_label.astype(np.float32)

        # With 60% probability, replace the background pixels in the input RGB image with real background
        if self.rng.rand() < 0.6:
            msk_back = (labels <= 0).astype(rgb.dtype)  # Create mask for background pixels
            msk_back = msk_back[:, :, None]  # Expand dimensions for broadcasting
            rgb = rgb * (msk_back == 0).astype(rgb.dtype) + back * msk_back  # Blend background

        # Replace background depth values where depth mask is invalid
        dpt = dpt * (dpt_msk > 0).astype(dpt.dtype) + \
              dpt_back * (dpt_msk <= 0).astype(dpt.dtype)

        return rgb, dpt

    def dpt_2_pcld(self, dpt, cam_scale, K):
        if len(dpt.shape) > 2:
            dpt = dpt[:, :, 0]
        dpt = dpt.astype(np.float32) / cam_scale
        msk = (dpt > 1e-8).astype(np.float32)
        row = (self.ymap - K[0][2]) * dpt / K[0][0]
        col = (self.xmap - K[1][2]) * dpt / K[1][1]
        dpt_3d = np.concatenate(
            (row[..., None], col[..., None], dpt[..., None]), axis=2
        )
        dpt_3d = dpt_3d * msk[:, :, None]
        return dpt_3d

    def get_item(self, item_name):
        # Check if the item is a .pkl file (rendered or fused data)
        if "pkl" in item_name:
            data = pkl.load(open(item_name, "rb"))
            dpt_mm = data['depth'] * 1000.
            # Extract RGB, mask labels, camera intrinsic matrix, and pose
            rgb = data['rgb']
            labels = data['mask']
            K = data['K']
            RT = data['RT']
            # Identify the type of rendered data (fused or normal render)
            rnd_typ = data['rnd_typ']

            # Process labels based on the type of data
            if rnd_typ == "fuse":
                labels = (labels == self.cls_id).astype("uint8")  # Only keep labels matching cls_id
            else:
                labels = (labels > 0).astype("uint8")  # Convert labels to binary mask
        else:
            # Load real data from depth, mask, and RGB images
            with Image.open(os.path.join(self.cls_root, f"depth/{item_name}.png")) as di:
                dpt_mm = np.array(di)

            with Image.open(os.path.join(self.cls_root, f"mask/{item_name}.png")) as li:
                labels = np.array(li)
                labels = (labels > 0).astype("uint8")  # Convert to binary mask

            with Image.open(os.path.join(self.cls_root, f"rgb/{item_name}.png")) as ri:
                if self.add_noise:
                    ri = self.trancolor(ri)  # Apply color transformation if noise is enabled
                rgb = np.array(ri)[:, :, :3]  # Extract RGB channels

            # Retrieve the corresponding metadata for the item
            meta = self.meta_lst[int(item_name)]

            # Special handling for class ID 2 (loop through metadata to find the right object)
            if self.cls_id == 2:
                for i in range(len(meta)):
                    if meta[i]['obj_id'] == 2:
                        meta = meta[i]
                        break
            else:
                meta = meta[0]  # Otherwise, take the first metadata entry

            # Extract camera pose (rotation and translation) and reshape R
            R = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
            T = np.array(meta['cam_t_m2c']) / 1000.0  # Convert translation to meters
            RT = np.concatenate((R, T[:, None]), axis=1)  # Create transformation matrix

            rnd_typ = 'real'  # Mark data type as real
            K = self.config.intrinsic_matrix["test"]  # Load intrinsic matrix

        cam_scale = 1000.0  # Scaling factor for depth

        # Ensure labels are 2D by extracting a single channel if needed
        if len(labels.shape) > 2:
            labels = labels[:, :, 0]

        rgb_labels = labels.copy()  # Keep a copy of the labels for RGB processing

        # # Add noise to rendered data (not real data) and blend with real backgrounds
        # if self.add_noise and rnd_typ != 'real':
        #     if rnd_typ == 'render' or self.rng.rand() < 0.8:
        #         rgb = self.rgb_add_noise(rgb)
        #         rgb_labels = labels.copy()
        #         msk_dp = dpt_mm > 1e-6  # Mask for valid depth pixels
        #         # rgb, dpt_mm = self.add_real_back(rgb, rgb_labels, dpt_mm, msk_dp)
        #
        #         # Additional noise augmentation with some probability
        #         if self.rng.rand() > 0.8:
        #             rgb = self.rgb_add_noise(rgb)
        #
        # # Display blendered rgb map if debugging mode is enabled
        # if self.DEBUG:
        #     show_real_back_map = rgb
        #     imshow("blender_rgb_map", show_real_back_map)
        #     waitKey(0)

        # Convert depth map to 16-bit unsigned integer format for further processing
        dpt_mm = dpt_mm.copy().astype(np.uint16)

        # Compute surface normal map from depth image using normalSpeed algorithm
        nrm_map = normalSpeed.depth_normal(
            dpt_mm, K[0][0], K[1][1], 5, 2000, 20, False
        )

        # Display normal map if debugging mode is enabled
        if self.DEBUG:
            show_nrm_map = ((nrm_map + 1.0) * 127).astype(np.uint8)
            imshow("nrm_map1", show_nrm_map)
            waitKey(0)

        # Convert depth from millimeters to meters
        dpt_m = dpt_mm.astype(np.float32) / cam_scale

        # Convert depth map to 3D point cloud
        dpt_xyz = self.dpt_2_pcld(dpt_m, 1.0, K)
        dpt_xyz[np.isnan(dpt_xyz)] = 0.0  # Replace NaN values with 0
        dpt_xyz[np.isinf(dpt_xyz)] = 0.0  # Replace infinite values with 0

        # Create a mask for valid depth values
        msk_dp = dpt_mm > 1e-6
        choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)

        # Ensure there are enough valid depth points
        if len(choose) < 400:
            return None

        # Generate a sequential index list
        choose_2 = np.array([i for i in range(len(choose))])
        if len(choose_2) < 400:
            return None

        # Downsample point selection if too many points exist
        if len(choose_2) > self.config.n_sample_points:
            c_mask = np.zeros(len(choose_2), dtype=int)
            c_mask[:self.config.n_sample_points] = 1
            np.random.shuffle(c_mask)
            choose_2 = choose_2[c_mask.nonzero()]
        else:
            choose_2 = np.pad(choose_2, (0, self.config.n_sample_points - len(choose_2)), 'wrap')

        # Select valid depth points
        choose = np.array(choose)[choose_2]

        # Shuffle selected points
        sf_idx = np.arange(choose.shape[0])
        np.random.shuffle(sf_idx)
        choose = choose[sf_idx]

        # Extract corresponding 3D points, RGB values, and normal vectors
        cld = dpt_xyz.reshape(-1, 3)[choose, :]
        rgb_pt = rgb.reshape(-1, 3)[choose, :].astype(np.float32)
        nrm_pt = nrm_map[:, :, :3].reshape(-1, 3)[choose, :]
        labels_pt = labels.flatten()[choose]
        choose = np.array([choose])

        # Concatenate 3D point cloud, RGB values, and normal vectors
        cld_rgb_nrm = np.concatenate((cld, rgb_pt, nrm_pt), axis=1).transpose(1, 0)

        # Retrieve ground-truth pose information
        RTs, kp3ds, ctr3ds, cls_ids, kp_targ_ofst, ctr_targ_ofst = self.get_pose_gt_info(
            cld, labels_pt, RT
        )

        # Compute depth and normal maps for the entire image
        h, w = rgb_labels.shape
        dpt_6c = np.concatenate((dpt_xyz, nrm_map[:, :, :3]), axis=2).transpose(2, 0, 1)
        rgb = np.transpose(rgb, (2, 0, 1))  # Convert RGB from HWC to CHW format

        # Multi-scale depth-to-point-cloud mapping
        xyz_lst = [dpt_xyz.transpose(2, 0, 1)]  # Store depth-to-point cloud map
        msk_lst = [dpt_xyz[2, :, :] > 1e-8]

        for i in range(3):
            scale = pow(2, i + 1)
            nh, nw = h // pow(2, i + 1), w // pow(2, i + 1)
            ys, xs = np.mgrid[:nh, :nw]
            xyz_lst.append(xyz_lst[0][:, ys * scale, xs * scale])
            msk_lst.append(xyz_lst[-1][2, :, :] > 1e-8)

        sr2dptxyz = {
            pow(2, ii): item.reshape(3, -1).transpose(1, 0)
            for ii, item in enumerate(xyz_lst)
        }

        # Downsampling configurations
        rgb_ds_sr = [4, 8, 8, 8]
        n_ds_layers = 4
        pcld_sub_s_r = [4, 4, 4, 4]
        inputs = {}

        # Downsampling stage
        for i in range(n_ds_layers):
            nei_idx = DP.knn_search(
                cld[None, ...], cld[None, ...], 16
            ).astype(np.int32).squeeze(0)
            sub_pts = cld[:cld.shape[0] // pcld_sub_s_r[i], :]
            pool_i = nei_idx[:cld.shape[0] // pcld_sub_s_r[i], :]
            up_i = DP.knn_search(
                sub_pts[None, ...], cld[None, ...], 1
            ).astype(np.int32).squeeze(0)

            inputs['cld_xyz%d' % i] = cld.astype(np.float32).copy()
            inputs['cld_nei_idx%d' % i] = nei_idx.astype(np.int32).copy()
            inputs['cld_sub_idx%d' % i] = pool_i.astype(np.int32).copy()
            inputs['cld_interp_idx%d' % i] = up_i.astype(np.int32).copy()

            nei_r2p = DP.knn_search(
                sr2dptxyz[rgb_ds_sr[i]][None, ...], sub_pts[None, ...], 16
            ).astype(np.int32).squeeze(0)
            inputs['r2p_ds_nei_idx%d' % i] = nei_r2p.copy()

            nei_p2r = DP.knn_search(
                sub_pts[None, ...], sr2dptxyz[rgb_ds_sr[i]][None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['p2r_ds_nei_idx%d' % i] = nei_p2r.copy()
            cld = sub_pts

        # Upsampling stage
        n_up_layers = 3
        rgb_up_sr = [4, 2, 2]
        for i in range(n_up_layers):
            r2p_nei = DP.knn_search(
                sr2dptxyz[rgb_up_sr[i]][None, ...],
                inputs['cld_xyz%d' % (n_ds_layers - i - 1)][None, ...], 16
            ).astype(np.int32).squeeze(0)
            inputs['r2p_up_nei_idx%d' % i] = r2p_nei.copy()

            p2r_nei = DP.knn_search(
                inputs['cld_xyz%d' % (n_ds_layers - i - 1)][None, ...],
                sr2dptxyz[rgb_up_sr[i]][None, ...], 1
            ).astype(np.int32).squeeze(0)
            inputs['p2r_up_nei_idx%d' % i] = p2r_nei.copy()

        # Prepare final item dictionary for training
        item_dict = dict(
            rgb=rgb.astype(np.uint8),
            cld_rgb_nrm=cld_rgb_nrm.astype(np.float32),
            choose=choose.astype(np.int32),
            labels=labels_pt.astype(np.int32),
            dpt_map_m=dpt_m.astype(np.float32),
            RTs=RTs.astype(np.float32),
            kp_targ_ofst=kp_targ_ofst.astype(np.float32),
            ctr_targ_ofst=ctr_targ_ofst.astype(np.float32),
            cls_ids=cls_ids.astype(np.int32),
            ctr_3ds=ctr3ds.astype(np.float32),
            kp_3ds=kp3ds.astype(np.float32),
        )

        item_dict.update(inputs)
        if self.DEBUG:
            extra_d = dict(
                dpt_xyz_nrm=dpt_6c.astype(np.float32),  # [6, h, w]
                cam_scale=np.array([cam_scale]).astype(np.float32),
                K=K.astype(np.float32),
            )
            item_dict.update(extra_d)
            item_dict['normal_map'] = nrm_map[:, :, :3].astype(np.float32)
        return item_dict

    def get_pose_gt_info(self, cld, labels, RT):
        # Initialize arrays for storing ground truth pose information
        RTs = np.zeros((self.config.n_objects, 3, 4))  # Rotation-Translation matrices
        kp3ds = np.zeros((self.config.n_objects, self.config.n_keypoints, 3))  # Keypoint 3D locations
        ctr3ds = np.zeros((self.config.n_objects, 3))  # Object center 3D locations
        cls_ids = np.zeros((self.config.n_objects, 1))  # Object class IDs
        kp_targ_ofst = np.zeros((self.config.n_sample_points, self.config.n_keypoints, 3))  # Keypoint offsets
        ctr_targ_ofst = np.zeros((self.config.n_sample_points, 3))  # Center offsets

        # Iterate over object class IDs (currently hardcoded for class_id = 1)
        for i, cls_id in enumerate([1]):
            # Store the provided rotation-translation (RT) matrix
            RTs[i] = RT
            r = RT[:, :3]  # Extract rotation matrix (3x3)
            t = RT[:, 3]  # Extract translation vector (3,)

            # Compute the transformed object center
            ctr = self.bs_utils.get_ctr(self.cls_type, ds_type="test")[:, None]  # Get object center
            ctr = np.dot(ctr.T, r.T) + t  # Apply transformation
            ctr3ds[i, :] = ctr[0]  # Store transformed center

            # Get indices of points belonging to the current object class
            msk_idx = np.where(labels == cls_id)[0]

            # Compute center offset for each point
            target_offset = np.array(np.add(cld, -1.0 * ctr3ds[i, :]))  # Offset from center
            ctr_targ_ofst[msk_idx, :] = target_offset[msk_idx, :]  # Store offsets for valid points

            # Store class ID
            cls_ids[i, :] = np.array([1])

            # Determine keypoint selection strategy
            self.minibatch_per_epoch = len(self.all_lst) // self.config.mini_batch_size
            if self.config.n_keypoints == 8:
                kp_type = 'farthest'  # Use farthest point sampling for keypoints
            else:
                kp_type = 'farthest{}'.format(self.config.n_keypoints)  # Adjust based on number of keypoints

            # Get and transform keypoints using rotation and translation
            kps = self.bs_utils.get_kps(self.cls_type, kp_type=kp_type, ds_type='test')
            kps = np.dot(kps, r.T) + t  # Apply transformation
            kp3ds[i] = kps  # Store transformed keypoints

            # Compute keypoint offsets for each point
            target = []
            for kp in kps:
                target.append(np.add(cld, -1.0 * kp))  # Offset from each keypoint
            target_offset = np.array(target).transpose(1, 0, 2)  # Reshape to [npts, nkps, c]
            kp_targ_ofst[msk_idx, :, :] = target_offset[msk_idx, :, :]  # Store offsets for valid points

        # Return computed pose information
        return RTs, kp3ds, ctr3ds, cls_ids, kp_targ_ofst, ctr_targ_ofst

    def __len__(self):
        return len(self.all_lst)

    def __getitem__(self, idx):
        if self.dataset_name == 'train':
            item_name = self.real_syn_gen()
            data = self.get_item(item_name)
            while data is None:
                item_name = self.real_syn_gen()
                data = self.get_item(item_name)
            return data
        else:
            item_name = self.all_lst[idx]
            return self.get_item(item_name)


def main():
    # config.mini_batch_size = 1
    ds = {}
    cls = 'L2'
    ds['train'] = Dataset('train', cls, DEBUG=True)
    ds['test'] = Dataset('test', cls, DEBUG=True)
    idx = dict(
        train=0,
        val=0,
        test=0
    )
    while True:
        # for cat in ['val', 'test']:
        # for cat in ['test']:
        for cat in ['train']:
            datum = ds[cat].__getitem__(idx[cat])
            idx[cat] += 1
            K = datum['K']
            cam_scale = datum['cam_scale']
            rgb = datum['rgb'].transpose(1, 2, 0)[..., ::-1].copy()  # [...,::-1].copy()
            for i in range(22):
                pcld = datum['cld_rgb_nrm'][:3, :].transpose(1, 0).copy()
                p2ds = ds[cat].bs_utils.project_p3d(pcld, cam_scale, K)
                # rgb = ds[cat].bs_utils.draw_p2ds(rgb, p2ds)
                kp3d = datum['kp_3ds'][i]
                if kp3d.sum() < 1e-6:
                    break
                kp_2ds = ds[cat].bs_utils.project_p3d(kp3d, cam_scale, K)
                rgb = ds[cat].bs_utils.draw_p2ds(
                    rgb, kp_2ds, 3, ds[cat].bs_utils.get_label_color(datum['cls_ids'][i][0], mode=1)
                )
                ctr3d = datum['ctr_3ds'][i]
                ctr_2ds = ds[cat].bs_utils.project_p3d(ctr3d[None, :], cam_scale, K)
                rgb = ds[cat].bs_utils.draw_p2ds(
                    rgb, ctr_2ds, 4, (0, 0, 255)
                )
            imshow('{}_rgb'.format(cat), rgb)
            cmd = waitKey(0)
            if cmd == ord('q'):
                exit()
            else:
                continue


if __name__ == "__main__":
    main()
# vim: ts=4 sw=4 sts=4 expandtab
