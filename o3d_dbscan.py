# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import open3d as o3d
import numpy as np
import copy
from matplotlib import pyplot as plt

from math import sin, cos, pi

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


if __name__ == "__main__":

    raw = o3d.io.read_point_cloud("./workspace.pcd")
    env_cloud = np.asarray(raw.points)

    temp_cloud = []
    for i in range(env_cloud.shape[0]):
        # x is the depth direction in RealSense coordiante
        # if env_cloud[i][0] < 850/1000.0:
        if env_cloud[i][0] < 850/1000.0:
            temp_cloud.append([env_cloud[i][0], env_cloud[i][1], env_cloud[i][2]])

    # print(len(temp_cloud))
    # print(len(temp_cloud[0]))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(temp_cloud))

    downpcd = pcd.voxel_down_sample(voxel_size=0.01)

    # o3d.visualization.draw_geometries([downpcd])

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            downpcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))



    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    downpcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([downpcd])
