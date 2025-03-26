#!/usr/bin/env python3
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import os
import tqdm
import cv2
import torch
import argparse
import torch.nn as nn
import numpy as np
import pickle as pkl
import pyrealsense2 as rs
from common import Config, ConfigRandLA
from models.ffb6d import FFB6D
from datasets.test_ycb.test_ycb_dataset import Dataset as TEST_YCB_Dataset
from datasets.ycb.ycb_dataset import Dataset as YCB_Dataset
from datasets.linemod.linemod_dataset import Dataset as LM_Dataset
from utils.pvn3d_eval_utils_kpls import cal_frame_poses, cal_frame_poses_lm
from utils.basic_utils import Basic_Utils
try:
    from neupeak.utils.webcv2 import imshow, waitKey
except ImportError:
    from cv2 import imshow, waitKey

import normalSpeed
from models.RandLA.helper_tool import DataProcessing as DP

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument(
    "-checkpoint", type=str, default=None, help="Checkpoint to eval"
)
parser.add_argument(
    "-dataset", type=str, default="linemod",
    help="Target dataset, ycb or linemod. (linemod as default)."
)
parser.add_argument(
    "-debug", type=bool, default=False,
    help="whether debug mode."
)
parser.add_argument(
    "-cam_scalar", type=int, default=10000,
    help="by dividing with the cam_scalar, the depth map becomes meters unit"
)
parser.add_argument(
    "-cls", type=str, default="ape",
    help="Target object to eval in LineMOD dataset. (ape, benchvise, cam, can," +
    "cat, driller, duck, eggbox, glue, holepuncher, iron, lamp, phone)"
)
parser.add_argument(
    "-show", action='store_true', help="View from imshow or not."
)
args = parser.parse_args()

if args.dataset != "linemod":
    config = Config(ds_name=args.dataset)
else:
    config = Config(ds_name=args.dataset, cls_type=args.cls)
bs_utils = Basic_Utils(config)

def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system('mkdir -p {}'.format(fd))

def get_item(rgb,dpt_um):
    xmap = np.array([[j for i in range(640)] for j in range(480)])
    ymap = np.array([[i for i in range(640)] for j in range(480)])
    K = config.intrinsic_matrix['ycb_test']

    cam_scale = args.cam_scalar
    msk_dp = (dpt_um > 1e-6) & (dpt_um <= 3000.0)

    dpt_mm = (dpt_um.copy() / 10).astype(np.uint16)
    nrm_map = normalSpeed.depth_normal(
        dpt_mm, K[0][0], K[1][1], 5, 300, 20, False
    )
    if args.debug:
        show_nrm_map = ((nrm_map + 1.0) * 127).astype(np.uint8)
        imshow("nrm_map", show_nrm_map)
        waitKey()

    dpt_m = dpt_um.astype(np.float32) / cam_scale
    dpt_xyz = dpt_2_pcld(dpt_m, 1.0, K, ymap, xmap)

    # downsample the mask
    choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
    if len(choose) < 400:
        return None
    choose_2 = np.array([i for i in range(len(choose))])
    if len(choose_2) < 400:
        return None
    if len(choose_2) > config.n_sample_points:
        c_mask = np.zeros(len(choose_2), dtype=int)
        c_mask[:config.n_sample_points] = 1
        np.random.shuffle(c_mask)
        choose_2 = choose_2[c_mask.nonzero()]
    else:
        choose_2 = np.pad(choose_2, (0, config.n_sample_points - len(choose_2)), 'wrap')
    choose = np.array(choose)[choose_2]

    sf_idx = np.arange(choose.shape[0])
    np.random.shuffle(sf_idx)
    choose = choose[sf_idx]

    cld = dpt_xyz.reshape(-1, 3)[choose, :]
    rgb_pt = rgb.reshape(-1, 3)[choose, :].astype(np.float32)
    nrm_pt = nrm_map[:, :, :3].reshape(-1, 3)[choose, :]
    choose = np.array([choose])
    cld_rgb_nrm = np.concatenate((cld, rgb_pt, nrm_pt), axis=1).transpose(1, 0)

    h, w, channel = rgb.shape
    dpt_6c = np.concatenate((dpt_xyz, nrm_map[:, :, :3]), axis=2).transpose(2, 0, 1)
    rgb = np.transpose(rgb, (2, 0, 1))  # hwc2chw

    xyz_lst = [dpt_xyz.transpose(2, 0, 1)]  # c, h, w
    msk_lst = [dpt_xyz[2, :, :] > 1e-8]

    for i in range(3):
        scale = pow(2, i + 1)
        nh, nw = h // pow(2, i + 1), w // pow(2, i + 1)
        ys, xs = np.mgrid[:nh, :nw]
        xyz_lst.append(xyz_lst[0][:, ys * scale, xs * scale])
        msk_lst.append(xyz_lst[-1][2, :, :] > 1e-8)
    sr2dptxyz = {
        pow(2, ii): item.reshape(3, -1).transpose(1, 0) for ii, item in enumerate(xyz_lst)
    }
    sr2msk = {
        pow(2, ii): item.reshape(-1) for ii, item in enumerate(msk_lst)
    }

    rgb_ds_sr = [4, 8, 8, 8]
    n_ds_layers = 4
    pcld_sub_s_r = [4, 4, 4, 4]
    inputs = {}
    # DownSample stage
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

    show_rgb = rgb.transpose(1, 2, 0).copy()[:, :, ::-1]
    if args.debug:
        for ip, xyz in enumerate(xyz_lst):
            pcld = xyz.reshape(3, -1).transpose(1, 0)
            p2ds = bs_utils.project_p3d(pcld, cam_scale, K)
            print(show_rgb.shape, pcld.shape)
            srgb = bs_utils.paste_p2ds(show_rgb.copy(), p2ds, (0, 0, 255))
            imshow("rz_pcld_%d" % ip, srgb)
            p2ds = bs_utils.project_p3d(inputs['cld_xyz%d' % ip], cam_scale, K)
            srgb1 = bs_utils.paste_p2ds(show_rgb.copy(), p2ds, (0, 0, 255))
            imshow("rz_pcld_%d_rnd" % ip, srgb1)
            waitKey()

    item_dict = dict(
        rgb=rgb.astype(np.uint8),  # [c, h, w]
        cld_rgb_nrm=cld_rgb_nrm.astype(np.float32),  # [9, npts]
        choose=choose.astype(np.int32),  # [1, npts]
        dpt_map_m=dpt_m.astype(np.float32),  # [h, w]
        cam_scale=np.array([cam_scale]).astype(np.float32),
        K=K.astype(np.float32)
    )
    item_dict.update(inputs)

    item_dict = ensure_data_keys(item_dict)
    # Reshaping each term individually

    # rgb: (3, 480, 640) -> (1, 3, 480, 640)
    item_dict['rgb'] = torch.tensor(item_dict['rgb']).unsqueeze(0)

    # cld_rgb_nrm: (9, 12800) -> (1, 9, 12800)
    item_dict['cld_rgb_nrm'] = torch.tensor(item_dict['cld_rgb_nrm']).unsqueeze(0)

    # choose: (1, 12800) -> (1, 1, 12800)
    item_dict['choose'] = torch.tensor(item_dict['choose']).unsqueeze(1)

    # dpt_map_m: (480, 640) -> (1, 480, 640)
    item_dict['dpt_map_m'] = torch.tensor(item_dict['dpt_map_m']).unsqueeze(0)

    # cam_scale: (1,) -> no change needed (already correct)
    item_dict['cam_scale'] = torch.tensor(item_dict['cam_scale'])

    # K: (3, 3) -> (1, 3, 3)
    item_dict['K'] = torch.tensor(item_dict['K']).unsqueeze(0)

    # cld_xyz0: (12800, 3) -> (1, 12800, 3)
    item_dict['cld_xyz0'] = torch.tensor(item_dict['cld_xyz0']).unsqueeze(0)

    # cld_nei_idx0: (12800, 16) -> (1, 12800, 16)
    item_dict['cld_nei_idx0'] = torch.tensor(item_dict['cld_nei_idx0']).unsqueeze(0)

    # cld_sub_idx0: (3200, 16) -> (1, 3200, 16)
    item_dict['cld_sub_idx0'] = torch.tensor(item_dict['cld_sub_idx0']).unsqueeze(0)

    # cld_interp_idx0: (12800, 1) -> (1, 12800, 1)
    item_dict['cld_interp_idx0'] = torch.tensor(item_dict['cld_interp_idx0']).unsqueeze(0)

    # r2p_ds_nei_idx0: (3200, 16) -> (1, 3200, 16)
    item_dict['r2p_ds_nei_idx0'] = torch.tensor(item_dict['r2p_ds_nei_idx0']).unsqueeze(0)

    # p2r_ds_nei_idx0: (19200, 1) -> (1, 19200, 1)
    item_dict['p2r_ds_nei_idx0'] = torch.tensor(item_dict['p2r_ds_nei_idx0']).unsqueeze(0)

    # cld_xyz1: (3200, 3) -> (1, 3200, 3)
    item_dict['cld_xyz1'] = torch.tensor(item_dict['cld_xyz1']).unsqueeze(0)

    # cld_nei_idx1: (3200, 16) -> (1, 3200, 16)
    item_dict['cld_nei_idx1'] = torch.tensor(item_dict['cld_nei_idx1']).unsqueeze(0)

    # cld_sub_idx1: (800, 16) -> (1, 800, 16)
    item_dict['cld_sub_idx1'] = torch.tensor(item_dict['cld_sub_idx1']).unsqueeze(0)

    # cld_interp_idx1: (3200, 1) -> (1, 3200, 1)
    item_dict['cld_interp_idx1'] = torch.tensor(item_dict['cld_interp_idx1']).unsqueeze(0)

    # r2p_ds_nei_idx1: (800, 16) -> (1, 800, 16)
    item_dict['r2p_ds_nei_idx1'] = torch.tensor(item_dict['r2p_ds_nei_idx1']).unsqueeze(0)

    # p2r_ds_nei_idx1: (4800, 1) -> (1, 4800, 1)
    item_dict['p2r_ds_nei_idx1'] = torch.tensor(item_dict['p2r_ds_nei_idx1']).unsqueeze(0)

    # cld_xyz2: (800, 3) -> (1, 800, 3)
    item_dict['cld_xyz2'] = torch.tensor(item_dict['cld_xyz2']).unsqueeze(0)

    # cld_nei_idx2: (800, 16) -> (1, 800, 16)
    item_dict['cld_nei_idx2'] = torch.tensor(item_dict['cld_nei_idx2']).unsqueeze(0)

    # cld_sub_idx2: (200, 16) -> (1, 200, 16)
    item_dict['cld_sub_idx2'] = torch.tensor(item_dict['cld_sub_idx2']).unsqueeze(0)

    # cld_interp_idx2: (800, 1) -> (1, 800, 1)
    item_dict['cld_interp_idx2'] = torch.tensor(item_dict['cld_interp_idx2']).unsqueeze(0)

    # r2p_ds_nei_idx2: (200, 16) -> (1, 200, 16)
    item_dict['r2p_ds_nei_idx2'] = torch.tensor(item_dict['r2p_ds_nei_idx2']).unsqueeze(0)

    # p2r_ds_nei_idx2: (4800, 1) -> (1, 4800, 1)
    item_dict['p2r_ds_nei_idx2'] = torch.tensor(item_dict['p2r_ds_nei_idx2']).unsqueeze(0)

    # cld_xyz3: (200, 3) -> (1, 200, 3)
    item_dict['cld_xyz3'] = torch.tensor(item_dict['cld_xyz3']).unsqueeze(0)

    # cld_nei_idx3: (200, 16) -> (1, 200, 16)
    item_dict['cld_nei_idx3'] = torch.tensor(item_dict['cld_nei_idx3']).unsqueeze(0)

    # cld_sub_idx3: (50, 16) -> (1, 50, 16)
    item_dict['cld_sub_idx3'] = torch.tensor(item_dict['cld_sub_idx3']).unsqueeze(0)

    # cld_interp_idx3: (200, 1) -> (1, 200, 1)
    item_dict['cld_interp_idx3'] = torch.tensor(item_dict['cld_interp_idx3']).unsqueeze(0)

    # r2p_ds_nei_idx3: (50, 16) -> (1, 50, 16)
    item_dict['r2p_ds_nei_idx3'] = torch.tensor(item_dict['r2p_ds_nei_idx3']).unsqueeze(0)

    # p2r_ds_nei_idx3: (4800, 1) -> (1, 4800, 1)
    item_dict['p2r_ds_nei_idx3'] = torch.tensor(item_dict['p2r_ds_nei_idx3']).unsqueeze(0)

    # r2p_up_nei_idx0: (200, 16) -> (1, 200, 16)
    item_dict['r2p_up_nei_idx0'] = torch.tensor(item_dict['r2p_up_nei_idx0']).unsqueeze(0)
    # p2r_up_nei_idx0: (19200, 1) -> (1, 19200, 1)
    item_dict['p2r_up_nei_idx0'] = torch.tensor(item_dict['p2r_up_nei_idx0']).unsqueeze(0)
    # r2p_up_nei_idx1: (800, 16) -> (1, 800, 16)
    item_dict['r2p_up_nei_idx1'] = torch.tensor(item_dict['r2p_up_nei_idx1']).unsqueeze(0)
    # p2r_up_nei_idx1: (76800, 1) -> (1, 76800, 1)
    item_dict['p2r_up_nei_idx1'] = torch.tensor(item_dict['p2r_up_nei_idx1']).unsqueeze(0)
    # r2p_up_nei_idx2: (3200, 16) -> (1, 3200, 16)
    item_dict['r2p_up_nei_idx2'] = torch.tensor(item_dict['r2p_up_nei_idx2']).unsqueeze(0)
    # p2r_up_nei_idx2: (76800, 1) -> (1, 76800, 1)
    item_dict['p2r_up_nei_idx2'] = torch.tensor(item_dict['p2r_up_nei_idx2']).unsqueeze(0)
    # labels: torch.Size([1, 3]) -> no change needed (already correct)
    # rgb_labels: torch.Size([1, 3]) -> no change needed (already correct)
    # RTs: torch.Size([1, 3]) -> no change needed (already correct)
    # kp_targ_ofst: torch.Size([1, 3]) -> no change needed (already correct)
    # ctr_targ_ofst: torch.Size([1, 3]) -> no change needed (already correct)
    # cls_ids: torch.Size([1, 3]) -> no change needed (already correct)
    # ctr_3ds: torch.Size([1, 3]) -> no change needed (already correct)
    # kp_3ds: torch.Size([1, 3]) -> no change needed (already correct)
    # cam_scale: torch.Size([1, 1]) -> no change needed (already correct)
    # K: torch.Size([3, 3]) -> (1, 3, 3) (already reshaped above)

    return item_dict


def ensure_data_keys(data):
    # Define the required keys
    required_keys = [
        'rgb', 'cld_rgb_nrm', 'choose', 'labels', 'rgb_labels', 'dpt_map_m', 'RTs',
        'kp_targ_ofst', 'ctr_targ_ofst', 'cls_ids', 'ctr_3ds', 'kp_3ds', 'cam_scale',
        'K', 'cld_xyz0', 'cld_nei_idx0', 'cld_sub_idx0', 'cld_interp_idx0',
        'r2p_ds_nei_idx0', 'p2r_ds_nei_idx0', 'cld_xyz1', 'cld_nei_idx1', 'cld_sub_idx1',
        'cld_interp_idx1', 'r2p_ds_nei_idx1', 'p2r_ds_nei_idx1', 'cld_xyz2',
        'cld_nei_idx2', 'cld_sub_idx2', 'cld_interp_idx2', 'r2p_ds_nei_idx2',
        'p2r_ds_nei_idx2', 'cld_xyz3', 'cld_nei_idx3', 'cld_sub_idx3', 'cld_interp_idx3',
        'r2p_ds_nei_idx3', 'p2r_ds_nei_idx3', 'r2p_up_nei_idx0', 'p2r_up_nei_idx0',
        'r2p_up_nei_idx1', 'p2r_up_nei_idx1', 'r2p_up_nei_idx2', 'p2r_up_nei_idx2'
    ]

    # Ensure all required keys are in the data dictionary
    for key in required_keys:
        if key not in data:
            data[key] = torch.zeros((1, 3), dtype=torch.float32).cuda()  # Default to zero for 3D points

    return data

def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):
    filename = "{}.pth.tar".format(filename)

    assert os.path.isfile(filename), "==> Checkpoint '{}' not found".format(filename)
    print("==> Loading from checkpoint '{}'".format(filename))
    try:
        checkpoint = torch.load(filename)
    except Exception:
        checkpoint = pkl.load(open(filename, "rb"))
    epoch = checkpoint.get("epoch", 0)
    it = checkpoint.get("it", 0.0)
    best_prec = checkpoint.get("best_prec", None)
    if model is not None and checkpoint["model_state"] is not None:
        ck_st = checkpoint['model_state']
        if 'module' in list(ck_st.keys())[0]:
            tmp_ck_st = {}
            for k, v in ck_st.items():
                tmp_ck_st[k.replace("module.", "")] = v
            ck_st = tmp_ck_st
        model.load_state_dict(ck_st)
    if optimizer is not None and checkpoint["optimizer_state"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    print("==> Done")
    return it, epoch, best_prec


def cal_view_pred_pose(model, data, epoch=0, obj_id=-1):
    model.eval()
    with torch.set_grad_enabled(False):
        cu_dt = {}
        # Ensure all required data keys are present
        # device = torch.device('cuda:{}'.format(args.local_rank))
        for key in data.keys():
            if data[key].dtype in [np.float32, np.uint8]:
                cu_dt[key] = torch.from_numpy(data[key].astype(np.float32)).cuda()
            elif data[key].dtype in [np.int32, np.uint32]:
                cu_dt[key] = torch.LongTensor(data[key].astype(np.int32)).cuda()
            elif data[key].dtype in [torch.uint8, torch.float32]:
                cu_dt[key] = data[key].float().cuda()
            elif data[key].dtype in [torch.int32, torch.int16]:
                cu_dt[key] = data[key].long().cuda()
        end_points = model(cu_dt)
        _, classes_rgbd = torch.max(end_points['pred_rgbd_segs'], 1)

        pcld = cu_dt['cld_rgb_nrm'][:, :3, :].permute(0, 2, 1).contiguous()
        if args.dataset == "ycb":
            pred_cls_ids, pred_pose_lst, _ = cal_frame_poses(
                pcld[0], classes_rgbd[0], end_points['pred_ctr_ofs'][0],
                end_points['pred_kp_ofs'][0], True, config.n_objects, True,
                None, None
            )
        elif args.dataset == "test_ycb":
            pred_cls_ids, pred_pose_lst, _ = cal_frame_poses(
                pcld[0], classes_rgbd[0], end_points['pred_ctr_ofs'][0],
                end_points['pred_kp_ofs'][0], True, config.n_objects, True,
                None, None
            )
        else:
            pred_pose_lst = cal_frame_poses_lm(
                pcld[0], classes_rgbd[0], end_points['pred_ctr_ofs'][0],
                end_points['pred_kp_ofs'][0], True, config.n_objects, False, obj_id
            )
            pred_cls_ids = np.array([[1]])

        np_rgb = cu_dt['rgb'].cpu().numpy().astype("uint8")[0].transpose(1, 2, 0).copy()
        if args.dataset == "ycb" or args.dataset == "test_ycb":
            np_rgb = np_rgb[:, :, ::-1].copy()
        ori_rgb = np_rgb.copy()
        for cls_id in cu_dt['cls_ids'][0].cpu().numpy():
            idx = np.where(pred_cls_ids == cls_id)[0]
            if len(idx) == 0:
                continue
            pose = pred_pose_lst[idx[0]]
            if args.dataset == "ycb" or args.dataset == "test_ycb":
                obj_id = int(cls_id[0])
            mesh_pts = bs_utils.get_pointxyz(obj_id, ds_type=args.dataset).copy()
            mesh_pts = np.dot(mesh_pts, pose[:, :3].T) + pose[:, 3]
            if args.dataset == "ycb":
                K = config.intrinsic_matrix["ycb_K1"]
            elif args.dataset == "test_ycb":
                K = config.intrinsic_matrix["ycb_test"]
            else:
                K = config.intrinsic_matrix["linemod"]
            mesh_p2ds = bs_utils.project_p3d(mesh_pts, 1.0, K)
            color = bs_utils.get_label_color(obj_id, n_obj=22, mode=2)
            np_rgb = bs_utils.draw_p2ds(np_rgb, mesh_p2ds, color=color)
        vis_dir = os.path.join(config.log_eval_dir, "pose_vis")
        ensure_fd(vis_dir)
        f_pth = os.path.join(vis_dir, "{}.jpg".format(epoch))
        if args.dataset == 'ycb' or args.dataset == 'test_ycb':
            bgr = np_rgb
            ori_bgr = ori_rgb
        else:
            bgr = np_rgb[:, :, ::-1]
            ori_bgr = ori_rgb[:, :, ::-1]
        cv2.imwrite(f_pth, bgr)
        if args.show:
            imshow("projected_pose_rgb", bgr)
            imshow("original_rgb", ori_bgr)
            waitKey()
    if epoch == 0:
        print("\n\nResults saved in {}".format(vis_dir))

def dpt_2_pcld(dpt, cam_scale, K, ymap, xmap):
    if len(dpt.shape) > 2:
        dpt = dpt[:, :, 0]
    dpt = dpt.astype(np.float32) / cam_scale
    msk = (dpt > 1e-8).astype(np.float32)
    row = (ymap - K[0][2]) * dpt / K[0][0]
    col = (xmap - K[1][2]) * dpt / K[1][1]
    dpt_3d = np.concatenate(
        (row[..., None], col[..., None], dpt[..., None]), axis=2
    )
    dpt_3d = dpt_3d * msk[:, :, None]
    return dpt_3d

def main():
    # Initialize RealSense camera pipeline
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(rs_config)

    # Initialize the model
    if args.dataset == "ycb":
        test_ds = YCB_Dataset('test')
        obj_id = -1
    elif args.dataset == "test_ycb":
        test_ds = TEST_YCB_Dataset('test')
        obj_id = -1
    else:
        test_ds = LM_Dataset('test', cls_type=args.cls)
        obj_id = config.lm_obj_dict[args.cls]

    rndla_cfg = ConfigRandLA
    model = FFB6D(
        n_classes=config.n_objects, n_pts=config.n_sample_points, rndla_cfg=rndla_cfg,
        n_kps=config.n_keypoints
    )
    model.cuda()

    # load status from checkpoint
    if args.checkpoint is not None:
        load_checkpoint(
            model, None, filename=args.checkpoint[:-8]
        )

    # Start capturing frames
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # get and generate input
            color_image = np.asanyarray(color_frame.get_data()).astype(np.float32)
            dpt_um = np.asanyarray(depth_frame.get_data()).astype(np.float32)
            input = get_item(color_image,dpt_um)

            cal_view_pred_pose(model, input, epoch=0, obj_id=obj_id)

            # Show the images if necessary
            if args.show:
                imshow("RGB Image", color_image)
                waitKey(1)
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()

# vim: ts=4 sw=4 sts=4 expandtab
