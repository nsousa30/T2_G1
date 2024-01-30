#!/usr/bin/env python3

import open3d as o3d  

def main():
    # cloud = o3d.io.read_point_cloud("apple_1_1_1.pcd") # Read point cloud
    cloud = o3d.io.read_point_cloud("./data/factory.ply") # Read point cloud
    o3d.visualization.draw_geometries([cloud])    # Visualize point cloud      

if __name__ == "__main__":
    main()