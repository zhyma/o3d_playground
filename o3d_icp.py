# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import open3d as o3d
import numpy as np
import copy

from math import sin, cos, pi

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


if __name__ == "__main__":
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

    # load environment point cloud and extract the foreground
    raw = o3d.io.read_point_cloud("./workspace.pcd")
    env_cloud = np.asarray(raw.points)

    temp_cloud = []
    for i in range(env_cloud.shape[0]):
        # x is the depth direction in RealSense coordiante
        if env_cloud[i][0] < 600/1000.0:
            temp_cloud.append([env_cloud[i][0], env_cloud[i][1], env_cloud[i][2]])

    foreground_pcd = o3d.geometry.PointCloud()
    foreground_pcd.points = o3d.utility.Vector3dVector(np.asarray(temp_cloud))


    target = foreground_pcd
    source = rod_pcd
    threshold = 0.02
    #trans_init = np.asarray(
    #            [[1.0, 0, 0,  500/1000.0],
    #            [0, 1.0, 0,  -100/1000.0],
    #            [0, 0,  1.0, 50/1000.0],
    #            [0.0, 0.0, 0.0, 1.0]])
    trans_init = np.asarray(
                [[1.0, 0, 0,  500/1000.0],
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
    draw_registration_result(source, target, reg_p2p.transformation)

    # print("Apply point-to-plane ICP")

    # target.estimate_normals()
    # reg_p2l = o3d.pipelines.registration.registration_icp(source, target, threshold, trans_init,
    #         o3d.pipelines.registration.TransformationEstimationPointToPlane())
    # print(reg_p2l)
    # print("Transformation is:")
    # print(reg_p2l.transformation)
    # print("")
    # draw_registration_result(source, target, reg_p2l.transformation)
