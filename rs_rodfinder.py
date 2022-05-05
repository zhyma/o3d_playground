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

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0., 0., 0.])

    raw = o3d.io.read_point_cloud("./workspace.pcd")
    env_cloud = np.asarray(raw.points)
    # o3d.visualization.draw_geometries([coord, raw])

    temp_cloud = []
    for i in range(env_cloud.shape[0]):
        # x is the depth direction in RealSense coordiante
        # if env_cloud[i][0] < 850/1000.0:
        if env_cloud[i][0] < 850/1000.0:
            temp_cloud.append([env_cloud[i][0], env_cloud[i][1], env_cloud[i][2]])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(temp_cloud))

    # o3d.visualization.draw_geometries([coord, pcd])

    ## Down sample to reduce the computatioinal load
    downpcd = pcd.voxel_down_sample(voxel_size=0.005)

    # print(len(downpcd.points))
    # o3d.visualization.draw_geometries([downpcd])

    ## Apply DBSCAN here
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            downpcd.cluster_dbscan(eps=0.05, min_points=10, print_progress=True))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    downpcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # o3d.visualization.draw_geometries([coord, downpcd])

    # first value: number of points assigned to the label
    # second value: x distance (depth) associated with this label
    dist_sum = np.zeros(((max_label+1), 2))
    
    for idx, val in enumerate(labels):
        dist_sum[val,0] += 1
        dist_sum[val,1] += downpcd.points[idx][0]

    selected = 0
    min_dist = 10.0e6
    for i in range(max_label+1):
        if (dist_sum[i,0] > 0):
            dist = dist_sum[i,1]/dist_sum[i,0]
            # print(dist)
            if dist < min_dist:
                selected = i
                min_dist = dist

    print("select label", selected, ", with distance ", min_dist)
    selected_pcd = o3d.geometry.PointCloud()
    selected_mat = np.zeros((int(dist_sum[selected,0]), 3))
    cnt = 0
    for idx, val in enumerate(labels):
        if val == selected:
            selected_mat[cnt,0] = downpcd.points[idx][0]
            selected_mat[cnt,1] = downpcd.points[idx][1]
            selected_mat[cnt,2] = downpcd.points[idx][2]
            cnt += 1

    selected_pcd.points = o3d.utility.Vector3dVector(np.asarray(selected_mat))
    # o3d.visualization.draw_geometries([coord, selected_pcd])

    cluster_center = [0.0, 0.0, 0.0]
    for i in range(len(selected_pcd.points)):
        cluster_center[0] += selected_pcd.points[i][0]
        cluster_center[1] += selected_pcd.points[i][1]
        cluster_center[2] += selected_pcd.points[i][2]

    for i in range(3):
        cluster_center[i] = cluster_center[i]/len(selected_pcd.points)

    # create rod template
    t = 60
    r = 20
    l = 200
    obj_np_cloud = np.zeros((t*l,3))
    for il in range(l):
        for it in range(t):
            idx = il*t+it

            obj_np_cloud[idx][2] = r*cos(it*pi/t)/1000.0
            obj_np_cloud[idx][0] = -r*sin(it*pi/t)/1000.0
            obj_np_cloud[idx][1] = il/1000.0

    rod_pcd = o3d.geometry.PointCloud()
    rod_pcd.points = o3d.utility.Vector3dVector(obj_np_cloud)

    target = selected_pcd
    source = rod_pcd
    threshold = 0.02
    # trans_init = np.asarray(
    #             [[1.0, 0, 0,  cluster_center[0]],
    #             [0, 1.0, 0,  cluster_center[1]],
    #             [0, 0,  1.0, cluster_center[2]],
    #             [0.0, 0.0, 0.0, 1.0]])
    trans_init = np.asarray(
                [[1.0, 0, 0,  cluster_center[0]],
                [0, 1.0, 0,  -100/1000.0],
                [0, 0,  1.0, 0],
                [0.0, 0.0, 0.0, 1.0]])
    
    draw_registration_result(source, target, trans_init)
    #print("Initial alignment")
    #evaluation = o3d.pipelines.registration.evaluate_registration(source, target,
    #        threshold, trans_init)
    #print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    # draw_registration_result(source, target, reg_p2p.transformation)
    draw_registration_result(source, downpcd, reg_p2p.transformation)


