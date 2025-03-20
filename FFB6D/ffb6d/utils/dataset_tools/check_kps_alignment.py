import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the .obj file
mesh = trimesh.load("example_mesh/L2/L2.obj")
keypoints = np.loadtxt('L2/L2_ORB_fps.txt')
# Sample N points randomly from the mesh surface
N = 5000
sampled_points, _ = trimesh.sample.sample_surface(mesh, N)

# Convert to NumPy
point_cloud_np = np.array(sampled_points)

# Plot using Matplotlib
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(point_cloud_np[:, 0], point_cloud_np[:, 1], point_cloud_np[:, 2], c='blue', s=1)
ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], c='r', s=20)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Sampled Point Cloud from OBJ")
plt.show()
