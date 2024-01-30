#!/usr/bin/env python3

import os as os
import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN

def load_point_cloud(file_path):
    return o3d.io.read_point_cloud(file_path)

def segment_plane(point_cloud, distance_threshold, ransac_n, num_iterations):
    plane_model, inliers = point_cloud.segment_plane(distance_threshold, ransac_n, num_iterations)
    return inliers, plane_model

def remove_floor_and_segment_table(point_cloud, distance_threshold=0.02, ransac_n=3, num_iterations=1000):
    # First, segment out the largest plane (likely the floor)
    floor_inliers, _ = segment_plane(point_cloud, distance_threshold, ransac_n, num_iterations)
    floor_removed = point_cloud.select_by_index(floor_inliers, invert=True) # Point cloud with the floor removed to be used in the second segmentation
    floor_pcd = point_cloud.select_by_index(floor_inliers) # Point cloud of the floor

    # Next, segment the table from the remaining points
    table_inliers, _ = segment_plane(floor_removed, distance_threshold, ransac_n, num_iterations)
    table_cloud = floor_removed.select_by_index(table_inliers) # Point cloud of the table

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

def align_z_axis_to_normal(point_cloud, new_z_axis):
    # Ensure the normals are computed
    if not point_cloud.has_normals():
        point_cloud.estimate_normals()

    # Calculate the rotation matrix
    current_z_axis = [0, 0, -1]  # Current Z-axis
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(np.cross(current_z_axis, new_z_axis))

    # Rotate the point cloud
    point_cloud.rotate(rotation_matrix, center=(0, 0, 0))
    return point_cloud

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


def main():
    # for file_index in range(1, 15):
        file_index = 5
        file_prefix = f"{file_index:02}"
        file_path = f'./data/{file_prefix}.ply'
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
        output_directory = "output_point_clouds/"
        os.makedirs(output_directory, exist_ok=True)

        for label in unique_labels:
            # Extract points belonging to the current cluster
            object_points = points[labels == label]

            # Create a new point cloud for the object
            object_point_cloud = o3d.geometry.PointCloud()
            object_point_cloud.points = o3d.utility.Vector3dVector(object_points)

            # Save the object's point cloud with the naming convention
            object_filename = f"{file_prefix}_{label:02}.ply"
            output_path = os.path.join(output_directory, object_filename)
            o3d.io.write_point_cloud(output_path, object_point_cloud)

        # Create a coordinate frame (referential) at the origin (size parameter adjusts the axis lengths)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

        # # Visualize the point cloud with the coordinate frame
        o3d.visualization.draw_geometries([object_point_cloud, coordinate_frame, aabb,original_point_cloud],
                                        window_name="Point Cloud with Coordinate Frame",
                                        width=800,
                                        height=600,
                                        left=50,
                                        top=50)

if __name__ == "__main__":
    main()