import pickle
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import json

def equalaxis(ax):
    # Ensure the axes have equal scale
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()

    # Find the min and max ranges across all axes
    all_limits = np.array([x_limits, y_limits, z_limits])
    min_val = all_limits[:, 0].min()
    max_val = all_limits[:, 1].max()

    # Set all axes to the same range
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_zlim(min_val, max_val)

rot_x = lambda deg: R.from_euler('x', np.radians(deg)).as_matrix()
rot_y = lambda deg: R.from_euler('y', np.radians(deg)).as_matrix()
rot_z = lambda deg: R.from_euler('z', np.radians(deg)).as_matrix()
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange']

def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

cam_config = load_config("../../../../blenderYCB/configs/config.json")["camera"]
R_mat = R.from_euler('xyz', np.radians(cam_config["location"]["rpy"])).as_matrix()  # 3x3 Rotation matrix
H_cam = np.eye(4)
H_cam[:3, :3] = R_mat  # Set rotation
H_cam[:3, 3] = cam_config["location"]["xyz"]     # Set translation
H_cam_inv = np.linalg.inv(H_cam)

# Load the dictionary from the pickle file
with open("datum.pkl", "rb") as pkl_file:
    datum_loaded = pickle.load(pkl_file)
rgb = datum_loaded["rgb"]
cld_rgb_nrm = datum_loaded["cld_rgb_nrm"]
cld = cld_rgb_nrm[:3, :].transpose(1, 0).copy()
kp_3ds = datum_loaded["kp_3ds"]
ctr_3ds = datum_loaded["ctr_3ds"]
RTs = datum_loaded["RTs"]
print(cld_rgb_nrm.shape)

# load pointcloud from camera
cld_inworldframe = ((rot_x(180) @ H_cam[:3,:3] @ cld.T) + H_cam[:3,3].reshape(3,1)).T
#cld_inworldframe = (rot_x(180) @ cld_inworldframe.T).T
cam_vecx = (H_cam[:3,:3] @ np.array([1,0,0]).T)*0.05
cam_vecy = (H_cam[:3,:3] @ np.array([0,1,0]).T)*0.05
cam_vecz = (H_cam[:3,:3] @ np.array([0,0,1]).T)*0.05

# Now datum_loaded is a dictionary
# Load the .obj file
base_link = trimesh.load("YCB_Video_Dataset/YCB_Video_Models/001_base_link/base_link.obj")
L2 = trimesh.load("YCB_Video_Dataset/YCB_Video_Models/models/002_L2/L2.obj")
L3 = trimesh.load("YCB_Video_Dataset/YCB_Video_Models/models/003_L3/L3.obj")
R2 = trimesh.load("YCB_Video_Dataset/YCB_Video_Models/models/004_R2/R2.obj")
R3 = trimesh.load("YCB_Video_Dataset/YCB_Video_Models/models/005_R3/R3.obj")
base_link_kps = np.loadtxt('001_base_link_8_kps.txt')
L2_kps = np.loadtxt('ycb_kps/002_L2_8_kps.txt')
L3_kps = np.loadtxt('ycb_kps/003_L3_8_kps.txt')
R2_kps = np.loadtxt('ycb_kps/004_R2_8_kps.txt')
R3_kps = np.loadtxt('ycb_kps/005_R3_8_kps.txt')
# Sample N points randomly from the mesh surface
N = 5000
base_link_cld_origin, _ = trimesh.sample.sample_surface(base_link, N)
L2_cld_origin, _ = trimesh.sample.sample_surface(L2, N)
L3_cld_origin, _ = trimesh.sample.sample_surface(L3, N)
R2_cld_origin, _ = trimesh.sample.sample_surface(R2, N)
R3_cld_origin, _ = trimesh.sample.sample_surface(R3, N)


# kps reverse
base_link_kps = (rot_x(90) @ base_link_kps.T).T
base_link_kps = (RTs[0,:,:3] @ base_link_kps.T + RTs[0,:,3].reshape(3,1)).T
L2_kps = (rot_x(90) @ L2_kps.T).T
L2_kps = (RTs[1,:,:3] @ L2_kps.T + RTs[1,:,3].reshape(3,1)).T
L3_kps = (rot_x(90) @ L3_kps.T).T
L3_kps = (RTs[2,:,:3] @ L3_kps.T + RTs[2,:,3].reshape(3,1)).T
R2_kps = (rot_x(90) @ R2_kps.T).T
R2_kps = (RTs[3,:,:3] @ R2_kps.T + RTs[3,:,3].reshape(3,1)).T
R3_kps = (rot_x(90) @ R3_kps.T).T
R3_kps = (RTs[4,:,:3] @ R3_kps.T + RTs[4,:,3].reshape(3,1)).T

# center


# Convert to NumPy
base_link_cld_origin = np.array(base_link_cld_origin)
L2_cld_origin = np.array(L2_cld_origin)
L3_cld_origin = np.array(L3_cld_origin)
R2_cld_origin = np.array(R2_cld_origin)
R3_cld_origin = np.array(R3_cld_origin)

# convert to blender coordinate
base_link_cld_origin = (rot_x(90) @ base_link_cld_origin.T).T
L2_cld_origin = (rot_x(90) @ L2_cld_origin.T).T
L3_cld_origin = (rot_x(90) @ L3_cld_origin.T).T
R2_cld_origin = (rot_x(90) @ R2_cld_origin.T).T
R3_cld_origin = (rot_x(90) @ R3_cld_origin.T).T


# transform obj
base_link_cld_2position = (RTs[0,:,:3] @ base_link_cld_origin.T + RTs[0,:,3].reshape(3,1)).T
L2_2position = (RTs[1,:,:3] @ L2_cld_origin.T + RTs[1,:,3].reshape(3,1)).T
L3_2position = (RTs[2,:,:3] @ L3_cld_origin.T + RTs[2,:,3].reshape(3,1)).T
R2_2position = (RTs[3,:,:3] @ R2_cld_origin.T + RTs[3,:,3].reshape(3,1)).T
R3_2position = (RTs[4,:,:3] @ R3_cld_origin.T + RTs[4,:,3].reshape(3,1)).T

# Plot using Matplotlib
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(base_link_cld_2position[:, 0], base_link_cld_2position[:, 1], base_link_cld_2position[:, 2], c=colors[5], s=1)
ax.scatter(L2_2position[:, 0], L2_2position[:, 1], L2_2position[:, 2], c=colors[5], s=1)
ax.scatter(L3_2position[:, 0], L3_2position[:, 1], L3_2position[:, 2], c=colors[5], s=1)
ax.scatter(R2_2position[:, 0], R2_2position[:, 1], R2_2position[:, 2], c=colors[5], s=1)
ax.scatter(R3_2position[:, 0], R3_2position[:, 1], R3_2position[:, 2], c=colors[5], s=1)

ax.scatter(base_link_kps[:, 0], base_link_kps[:, 1], base_link_kps[:, 2], c=colors[0], s=40)
ax.scatter(L2_kps[:, 0], L2_kps[:, 1], L2_kps[:, 2], c=colors[1], s=40)
ax.scatter(L3_kps[:, 0], L3_kps[:, 1], L3_kps[:, 2], c=colors[2], s=40)
ax.scatter(R2_kps[:, 0], R2_kps[:, 1], R2_kps[:, 2], c=colors[3], s=40)
ax.scatter(R3_kps[:, 0], R3_kps[:, 1], R3_kps[:, 2], c=colors[4], s=40)

ax.scatter(cld_inworldframe[:,0],cld_inworldframe[:,1],cld_inworldframe[:,2],c=colors[7], s=4)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("transformed cld in world frame")
equalaxis(ax)
plt.show(block=True)

# fig2 = plt.figure(figsize=(10, 10))
# ax2 = fig2.add_subplot(111, projection='3d')
# ax2.scatter(base_link_cld_origin[:, 0], base_link_cld_origin[:, 1], base_link_cld_origin[:, 2], c=colors[0], s=1)
# ax2.scatter(L2_cld_origin[:, 0], L2_cld_origin[:, 1], L2_cld_origin[:, 2], c=colors[1], s=1)
# ax2.scatter(L3_cld_origin[:, 0], L3_cld_origin[:, 1], L3_cld_origin[:, 2], c=colors[2], s=1)
# ax2.scatter(R2_cld_origin[:, 0], R2_cld_origin[:, 1], R2_cld_origin[:, 2], c=colors[3], s=1)
# ax2.scatter(R3_cld_origin[:, 0], R3_cld_origin[:, 1], R3_cld_origin[:, 2], c=colors[4], s=1)
# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
# ax2.set_zlabel('Z')
# ax2.set_title("original cld in world frame")
# equalaxis(ax2)
# plt.show()

#
# fig3 = plt.figure(figsize=(10, 10))
# ax3 = fig3.add_subplot(111, projection='3d')
# ax3.scatter(base_link_cld_origin[:, 0], base_link_cld_origin[:, 1], base_link_cld_origin[:, 2], c=colors[0], s=1)
# ax3.scatter(L2_cld_origin[:, 0], L2_cld_origin[:, 1], L2_cld_origin[:, 2], c=colors[1], s=1)
# ax3.scatter(L3_cld_origin[:, 0], L3_cld_origin[:, 1], L3_cld_origin[:, 2], c=colors[2], s=1)
# ax3.scatter(R2_cld_origin[:, 0], R2_cld_origin[:, 1], R2_cld_origin[:, 2], c=colors[3], s=1)
# ax3.scatter(R3_cld_origin[:, 0], R3_cld_origin[:, 1], R3_cld_origin[:, 2], c=colors[4], s=1)
# ax3.set_xlabel('X')
# ax3.set_ylabel('Y')
# ax3.set_zlabel('Z')
# ax3.set_title("original cld in blender world frame")
# equalaxis(ax3)
# plt.show()
