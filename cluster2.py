#!/usr/bin/env python3

import os
import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def load_point_cloud(file_path):
    return o3d.io.read_point_cloud(file_path)

def segment_plane(point_cloud, distance_threshold, ransac_n, num_iterations):
    plane_model, inliers = point_cloud.segment_plane(distance_threshold, ransac_n, num_iterations)
    return inliers, plane_model

def remove_floor_and_segment_table(point_cloud, distance_threshold=0.02, ransac_n=3, num_iterations=1000):
    # First, segment out the largest plane (likely the floor)
    floor_inliers, _ = segment_plane(point_cloud, distance_threshold, ransac_n, num_iterations)
    floor_removed = point_cloud.select_by_index(floor_inliers, invert=True)  # Point cloud with the floor removed to be used in the second segmentation
    floor_pcd = point_cloud.select_by_index(floor_inliers)  # Point cloud of the floor

    # Next, segment the table from the remaining points
    table_inliers, _ = segment_plane(floor_removed, distance_threshold, ransac_n, num_iterations)
    table_cloud = floor_removed.select_by_index(table_inliers)  # Point cloud of the table

    points_table = np.asarray(table_cloud.points)
    points_floor = np.asarray(floor_pcd.points)

    table_height = np.mean(points_table[:, 2])
    floor_height = np.mean(points_floor[:, 2])

    if table_height < floor_height:
        final_cloud = floor_pcd
    else:
        final_cloud = table_cloud
    return final_cloud

def color_points(point_cloud, color):
    colors = [color for i in range(len(point_cloud.points))]
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

def center_referential(point_cloud):
    # Center the referential with the centroid of the point cloud
    points = np.asarray(point_cloud.points)
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    point_cloud.points = o3d.utility.Vector3dVector(points_centered)

    # Calculate the final centroid for later use
    points_final = np.asarray(point_cloud.points)
    final_centroid = np.mean(points_final, axis=0)
    translation_vector = final_centroid - centroid

    # Example: Aligning Z-axis with the average normal of the point cloud
    point_cloud.estimate_normals()
    normals = np.asarray(point_cloud.normals)
    average_normal = np.mean(normals, axis=0)
    average_normal /= np.linalg.norm(average_normal)  # Normalize the average normal

    # Calculate the rotation matrix
    current_z_axis = [0, 0, -1]  # Current Z-axis
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(np.cross(current_z_axis, average_normal))

    return rotation_matrix, translation_vector

def denoise(point_cloud):
    # Estimate normals for DBSCAN
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Apply DBSCAN clustering
    labels = np.array(point_cloud.cluster_dbscan(eps=0.02, min_points=100, print_progress=True))

    max_label = labels.max()
    largest_cluster_index = None
    largest_cluster_size = 0

    # Iterate through each cluster to find the largest one
    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) > largest_cluster_size:
            largest_cluster_size = len(cluster_indices)
            largest_cluster_index = i

    # Select only the largest cluster
    if largest_cluster_index is not None:
        largest_cluster_indices = np.where(labels == largest_cluster_index)[0]
        point_cloud = point_cloud.select_by_index(largest_cluster_indices)

    return point_cloud

def calculate_aabb(point_cloud):
    aabb = point_cloud.get_axis_aligned_bounding_box()

    # Get the current bounds
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()

    # Modify the Z-coordinate of the maximum bound
    max_bound[2] += 0.5  # Extend one unit upwards

    # Create a new AABB with the modified bounds
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

    aabb.color = (1, 0, 0)  # RGB for red

    return aabb

def voxel_downsize(point_cloud, size):
    # Apply voxel grid downsampling
    downsampled_cloud = point_cloud.voxel_down_sample(size)

    return downsampled_cloud

def save_images_from_point_cloud(point_cloud, output_directory, file_prefix):
    os.makedirs(output_directory, exist_ok=True)

    # Ajuste a escala da imagem para garantir que todos os pontos são visíveis
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(point_cloud)
    vis.get_render_option().point_size = 1
    vis.get_render_option().background_color = np.asarray([1, 1, 1])  # fundo branco
    vis.run()
    vis.capture_screen_image(os.path.join(output_directory, f"{file_prefix}_point_cloud.png"), do_render=True)
    vis.destroy_window()

def visualize_saved_point_clouds(output_directory, file_prefix):
    ply_files = [f for f in os.listdir(output_directory) if f.endswith(".ply")]

    for ply_file in ply_files:
        ply_path = os.path.join(output_directory, ply_file)
        point_cloud = o3d.io.read_point_cloud(ply_path)

        # Visualize the point cloud
        o3d.visualization.draw_geometries([point_cloud], window_name=f"Nuvem de Pontos para {file_prefix}")

def main():
    file_index = 11
    file_prefix = f"{file_index:02}"
    file_path = '/home/alexandre/SAVI/T2_G1/data/11.ply'
    original_point_cloud = load_point_cloud(file_path)

    # Center the referential and align Z-axis and rotate the point cloud
    rotation_matrix, _ = center_referential(original_point_cloud)
    original_point_cloud = original_point_cloud.rotate(rotation_matrix, center=(0, 0, 0))

    # Remove the floor, segment the table
    point_cloud = remove_floor_and_segment_table(original_point_cloud)

    # Denoise
    point_cloud = denoise(point_cloud)

    # Color table points in red
    color_points(point_cloud, [1, 0, 0])  # Red color

    # Center the referential and align Z-axis and rotate the point cloud
    rotation_matrix, transfer_vector = center_referential(point_cloud)
    aligned_point_cloud = point_cloud.rotate(rotation_matrix, center=(0, 0, 0))

    aligned_original_point_cloud = original_point_cloud.rotate(rotation_matrix, center=(0, 0, 0))
    aligned_original_point_cloud.translate(transfer_vector)

    # Voxel downsize for ease of use
    aligned_point_cloud = voxel_downsize(point_cloud, 0.01)
    aligned_original_point_cloud = voxel_downsize(original_point_cloud, 0.01)

    # Create bounding box surrounding table
    aabb = calculate_aabb(aligned_point_cloud)

    # Crop point cloud to bounding box
    aligned_original_point_cloud = aligned_original_point_cloud.crop(aabb)

    # Segment plane of new point cloud
    new_point_cloud_inliers, _ = segment_plane(aligned_original_point_cloud, distance_threshold=0.05, ransac_n=3, num_iterations=1500)
    new_point_cloud = aligned_original_point_cloud.select_by_index(new_point_cloud_inliers)

    # Center the referential and align Z-axis and rotate the point cloud
    rotation_matrix, transfer_vector = center_referential(new_point_cloud)
    new_point_cloud = new_point_cloud.rotate(rotation_matrix, center=(0, 0, 0))

    aligned_original_point_cloud = original_point_cloud.rotate(rotation_matrix, center=(0, 0, 0))
    aligned_original_point_cloud.translate(transfer_vector)

    # Create bounding box surrounding table
    aabb = calculate_aabb(new_point_cloud)
    center = aabb.get_center()
    dimensions = aabb.get_max_bound() - aabb.get_min_bound()
    depth = dimensions[2]
    new_center = np.array([center[0], center[1], depth / 2.0])
    translation_vector = new_center - center
    aabb.translate(translation_vector)

    # Crop point cloud to bounding box
    aligned_original_point_cloud = aligned_original_point_cloud.crop(aabb)

    # Object clustering
    aligned_original_point_cloud_inliers, _ = segment_plane(aligned_original_point_cloud, distance_threshold=0.02, ransac_n=3, num_iterations=1000)
    object_point_cloud = aligned_original_point_cloud.select_by_index(aligned_original_point_cloud_inliers, invert=True)

    object_cloud_points = points = np.asarray(object_point_cloud.points)
    clustering = DBSCAN(eps=0.02, min_samples=100).fit(points)
    labels = clustering.labels_
    unique_labels = np.unique(labels[labels >= 0])

    # Save each object as an RGB image
    output_directory = "output_rgb_images/"
    os.makedirs(output_directory, exist_ok=True)

    for label in unique_labels:
        # Extract points belonging to the current cluster
        object_points = points[labels == label]

        # Create a new point cloud for the object
        object_point_cloud = o3d.geometry.PointCloud()
        object_point_cloud.points = o3d.utility.Vector3dVector(object_points)

# Salve imagens da nuvem de pontos
        save_images_from_point_cloud(original_point_cloud, "output_images/", file_prefix)

    # Visualize as nuvens de pontos salvas
    visualize_saved_point_clouds("output_point_clouds/", file_prefix)

if __name__ == "__main__":
    main()
