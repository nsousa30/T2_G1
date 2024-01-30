#!/usr/bin/env python3

import os as os
import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import math
from scipy import stats
from matplotlib import cm
from more_itertools import locate

view = {
    "class_name": "ViewTrajectory",
    "interval": 29,
    "is_loop": False,
    "trajectory":
        [
            {
                "boundingbox_max": [2.6540005122611348, 2.3321821423160629, 0.85104994623420782],
                "boundingbox_min": [-2.5261458770339673, -2.1656718060235378, -0.55877501755379944],
                "field_of_view": 60.0,
                "front": [0.75672239933786944, 0.34169632162348007, 0.55732830013316348],
                "lookat": [0.046395260625899069, 0.011783639768603466, -0.10144691776517496],
                "up": [-0.50476400916821107, -0.2363660920597864, 0.83026764695055955],
                "zoom": 0.30119999999999997
            }
        ],
    "version_major": 1,
    "version_minor": 0
}


thresholds = { "up": -0.1, "down": -0.55, "first_outliers": [150, 0.3], "second_outliers": [20, 0.05], "delta_norm": 0.05}


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


if __name__ == "__main__":
    # for file_index in range(1, 15):
    file_index = 3
    file_prefix = f"{file_index:02}"
    original_point_cloud = o3d.io.read_point_cloud(f'./data/{file_prefix}.ply')
    
    pcd_downsampled = original_point_cloud.voxel_down_sample(voxel_size=0.02)
    original_pcd = original_point_cloud.voxel_down_sample(voxel_size=0.02)
    
    # estimate normals
    pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_downsampled.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, -1]))
    
    points = np.asarray(pcd_downsampled.points)
    arr1 = np.round(pcd_downsampled.normals,decimals = 3)
    

    # Deleting normal vectores "too horizintal", to ensure mode vector being normal to the table top and not to the walls (case of images 5 to 8)------
    idx = 0
    while idx < len(arr1):
        if arr1[idx][0] < -0.6 or arr1[idx][0] > 0.6:
            arr1 = np.delete(arr1, idx, axis=0)
            points = np.delete(points, idx, axis=0)
        else:
            idx += 1

    # ---------------------------------------------------------------------------------------------------------


    vetor_moda = stats.mode(arr1, keepdims=True)

    # Deleting points from pointcloud that have its normal vector too far away from the mode normal vector.
    # We do this to filter the point cloud to just horizontal surfaces (like the table, and the floor)-------------
    idx = 0
    while idx < len(points):
        if np.linalg.norm(arr1[idx] - vetor_moda[0]) > thresholds["delta_norm"]:
            arr1 = np.delete(arr1, idx, axis=0)
            points = np.delete(points, idx, axis=0)
        else:
            idx += 1

    pcd_downsampled.points = o3d.utility.Vector3dVector(points)
    # ---------------------------------------------------------------------------------------------------------
    
    # Create transformation T1 only with rotation
    T1 = np.zeros((4, 4), dtype=float)

    # Add homogeneous coordinates
    T1[3, 3] = 1

    vetor_moda_simples = [vetor_moda[0][0][0], vetor_moda[0][0][1], vetor_moda[0][0][2]]
    
    print("vetor_moda_simples")
    print(vetor_moda_simples)
    
    # rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(np.cross([0, 0, 1], vetor_moda_simples))
    rotation_matrix = rotation_matrix_from_vectors([0, 0, 1], vetor_moda_simples)
    print("rotation_matrix")
    print(rotation_matrix)
    

    T1[0:3, 0:3] = rotation_matrix


    points = np.dot(points, rotation_matrix)  #colocar os pontos com as novas coordenadas, para filtrar valores de altura


    # Filtrar pontos pela sua altura (eixo z). So queremos ficar com a mesa (e alguns pontos residuais) ----------------
    chao = min(points[:][2])
    
    altura_minima = 0.3

    

    idx = 0
    while idx < len(points):

        if  points[idx][2] <thresholds["down"] or points[idx][2] > thresholds["up"]:
           points = np.delete(points, idx, axis=0)
        else:
           idx += 1

    points = np.dot(points, np.linalg.inv(rotation_matrix))

    pcd_downsampled.points = o3d.utility.Vector3dVector(points)

    # ---------------------------------------------------------------------------------------------------------------------
    
    
    #  Filter points in pointcloud to delete every points with few neibourghs. The goal is to have only the table top ------------------------------

    # print("Statistical oulier removal")
    # cl, ind = pcd_downsampled.remove_statistical_outlier(nb_neighbors=thresholds["first_outliers"][0], std_ratio=thresholds["first_outliers"][1])
    # inlier_cloud = cl.select_by_index(ind)

    print("Radius oulier removal")
    cl, ind = pcd_downsampled.remove_radius_outlier(nb_points=thresholds["first_outliers"][0], radius=thresholds["first_outliers"][1])
    # inlier_cloud = cl.select_by_index(ind)

   

    print("Radius oulier removal")
    cl, ind = cl.remove_radius_outlier(nb_points=thresholds["second_outliers"][0], radius=thresholds["second_outliers"][1])

    print("\nNumber of Points:")
    print(len(np.asarray(cl.points)))

    # ---------------------------------------------------------------------------------------------------------------------
    
    # Compute the centroid of pointcloud. The goal is to get the translation vector from world frame to table top center frame---------------------
    centroid = cl.get_center()
    print("centroid:")
    print(centroid)

    # ---------------------------------------------------------------------------------------------------------------------------------
    # Add a translation
    T1[0:3, 3] = [centroid[0], centroid[1], centroid[2]]

    # print("T1")
    # print(T1)
    # Create table ref system and apply transformation to it
    frame_table = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2, origin=np.array([0., 0., 0.]))

    frame_table = frame_table.transform(T1)

    frame_world = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))

    #Create crop near the table ---------------------------------------------------------------------------
    pcd_downsampled = pcd_downsampled.transform(np.linalg.inv(T1))
    original_pcd = original_pcd.transform(np.linalg.inv(T1))

    np_vertices = np.ndarray([8,3], dtype=float)
    new_vertices = np.ndarray([8,3], dtype=float)

    sx=sy=0.7
    sz_top=0.35
    sz_bottom = -0.05
    np_vertices[0,0:3] = [sx,sy,sz_top] 
    np_vertices[1,0:3] = [sx,-sy,sz_top] 
    np_vertices[2,0:3] = [-sx,-sy,sz_top] 
    np_vertices[3,0:3] = [-sx,sy,sz_top] 
    np_vertices[4,0:3] = [sx,sy,sz_bottom] 
    np_vertices[5,0:3] = [sx,-sy,sz_bottom] 
    np_vertices[6,0:3] = [-sx,-sy,sz_bottom] 
    np_vertices[7,0:3] = [-sx,sy,sz_bottom] 

    vertices = o3d.utility.Vector3dVector(np_vertices)  

    # bb_vertices_table_frame = np.dot(np_vertices,  np.linalg.inv(rotation_matrix)) 
    # for i in range(0, 8):
    #     new_vertices[i, 0:3] = bb_vertices_table_frame[i, 0:3] + centroid


    # bb_vertices_table_frame = np.sum(bb_vertices_table_frame, T1[0:3, 3]) 

    bbox=o3d.geometry.AxisAlignedBoundingBox.create_from_points(vertices)
    # bbox=o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(np_vertices))

    # bbox = bbox.transform(T1)
    # bbox=o3d.geometry.AxisAlignedBoundingBox.create_from_points(vertices)

    pcd_cropped = original_pcd.crop(bbox)


    #Plane segmentation --------------------

    plane_model, inlier_idxs = pcd_cropped.segment_plane(distance_threshold=0.02,ransac_n=3, num_iterations=100)

    a, b, c, d = plane_model
    pcd_table = pcd_cropped.select_by_index(inlier_idxs, invert=False)
    pcd_table.paint_uniform_color([1, 0, 0])

    pcd_objects = pcd_cropped.select_by_index(inlier_idxs, invert=True)

    #clustering --------------------------

    labels = pcd_objects.cluster_dbscan(eps=0.05, min_points=50, print_progress=True)

    print("Max label:", max(labels))

    group_idxs = list(set(labels))
    # group_idxs.remove(-1)  # remove last group because its the group on the unassigned points
    num_groups = len(group_idxs)
    colormap = cm.Pastel1(range(0, num_groups))

    pcd_separate_objects = []
    for group_idx in group_idxs:  # Cycle all groups, i.e.,

        group_points_idxs = list(locate(labels, lambda x: x == group_idx))

        pcd_separate_object = pcd_objects.select_by_index(group_points_idxs, invert=False)

        color = colormap[group_idx, 0:3]
        pcd_separate_object.paint_uniform_color(color)
        pcd_separate_objects.append(pcd_separate_object)

    #Save the image 
    
    # Save each separated object as a PLY file
    output_folder = "./output_point_clouds"
    os.makedirs(output_folder, exist_ok=True)

    for i, pcd_separate_object in enumerate(pcd_separate_objects):
        output_filename = os.path.join(output_folder, f"{file_prefix}_object_{i}.ply")
        o3d.io.write_point_cloud(output_filename, pcd_separate_object)
        print(f"Object {i} saved as {output_filename}")

    # Visualization ----------------------

  
    pcds_to_draw = [pcd_objects]   

    entities = []
    entities.append(bbox)
    entities.append(frame_table)
    entities.append(frame_world)
    entities.extend(pcds_to_draw)
    o3d.visualization.draw_geometries(entities,
                                      zoom=0.3412,
                                      front=view['trajectory'][0]['front'],
                                      lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'], point_show_normal=True)
    
    #see objects in alternating order

    ply_files = [f for f in os.listdir(output_folder) if f.endswith(".ply")]

    for ply_file in ply_files:
        ply_path = os.path.join(output_folder, ply_file)
        point_cloud = o3d.io.read_point_cloud(ply_path)

        # Visualize the point cloud
        o3d.visualization.draw_geometries([point_cloud], window_name=f"Nuvem de Pontos para {file_prefix}")

    # # Visualize the point cloud with the coordinate frame
    # o3d.visualization.draw_geometries(original_point_cloud,
    #                                 window_name="Point Cloud with Coordinate Frame",
    #                                 width=800,
    #                                 height=600,
    #                                 left=50,
    #                                 top=50)