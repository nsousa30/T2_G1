#!/usr/bin/env python3

import os as os
import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import math
from scipy import stats
from matplotlib import cm
from more_itertools import locate
import cv2
import colorsys
import argparse

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
                "zoom": 0.1
            }
        ],
    "version_major": 1,
    "version_minor": 0
}





thresholds = { "up": -0.1, "down": -0.55, "first_outliers": [150, 0.3], "second_outliers": [20, 0.05], "delta_norm": 0.05}

hsv_image_boundaries = { "chavena_amarela": { "H": [24, 30], "S": [164, 255], "V": [111, 180]},
                        "chapeu_branco": { "H": [31, 255], "S": [0, 44], "V": [168, 255]},
                        "chapeu_preto": { "H": [76, 155], "S": [12, 210], "V": [0, 30]},
                        "lata_verde": { "H": [35, 100], "S": [38, 255], "V": [72, 187]},
                        "chavena_branca": { "H": [21, 123], "S": [0, 45], "V": [123, 241]},
                        "taca_amarela": { "H": [17, 28], "S": [241, 255], "V": [77, 201]},
                        "tijela_branca": { "H": [31, 99], "S": [15, 201], "V": [49, 219]},
                        "cl_Castanho_branco": { "H": [11, 30], "S": [69, 255], "V": [83, 148]},}


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



def parse_arguments():
    parser = argparse.ArgumentParser(description=' With this program, you will detect each object in RGB-D scene',
                                     epilog="Go ahead! Choose your scene!")

    parser.add_argument('-sc', '--scene', help="Number of scene that you want to see.", default='12')

    return parser.parse_args()


if __name__ == "__main__":

    # --------------------------------------
    # Initialization
    # --------------------------------------
    args = parse_arguments()
    file_index = args.scene
    # for file_index in range(1, 15):
    # file_index = 12
    file_prefix = f"{int(file_index):02}"
    filename_rgb_image = f'rgbd_scenes_dataset/{file_prefix}-color.png'
    filename_depth_image = f'rgbd_scenes_dataset/{file_prefix}-depth.png'
    color_raw = o3d.io.read_image(filename_rgb_image)
    depth_raw = o3d.io.read_image(filename_depth_image)

    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_scale=5000, convert_rgb_to_intensity=False)

    print(type(color_raw))

    fx = 570
    fy = 570
    cx = 320
    cy = 240
    width = 640
    height = 480
    K = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, K)

    
    # --------------------------------------
    # Visualization
    # --------------------------------------

    entities = [point_cloud]

    o3d.visualization.draw_geometries(entities,
                                       zoom=0.3412,
                                       front=view['trajectory'][0]['front'],
                                       lookat=view['trajectory'][0]['lookat'],
                                       up=view['trajectory'][0]['up'],
                                       point_show_normal=False)

    # --------------------------------------
    # Termination
    # -------------------------------------- 
    
    pcd_downsampled = point_cloud.voxel_down_sample(voxel_size=0.02)
    original_pcd = point_cloud.voxel_down_sample(voxel_size=0.02)
    
    
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
    
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(np.cross([0, 0, 1], vetor_moda_simples))
    rotation_matrix = rotation_matrix_from_vectors([0, 0, 1], vetor_moda_simples)

    T1[0:3, 0:3] = rotation_matrix


    points = np.dot(points, rotation_matrix)  #colocar os pontos com as novas coordenadas, para filtrar valores de altura

    # ---------------------------------------------------------------------------------------------------------------------
    
    # Compute the centroid of pointcloud. The goal is to get the translation vector from world frame to table top center frame---------------------
    centroid = pcd_downsampled.get_center()

    # ---------------------------------------------------------------------------------------------------------------------------------
    # Add a translation
    T1[0:3, 3] = [centroid[0], centroid[1], centroid[2]]

    # Create table ref system and apply transformation to it
    frame_table = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2, origin=np.array([0., 0., 0.]))

    frame_table = frame_table.transform(T1)

    frame_world = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))

    #Create crop near the table ---------------------------------------------------------------------------
    pcd_downsampled = pcd_downsampled.transform(np.linalg.inv(T1))
    original_pcd = original_pcd.transform(np.linalg.inv(T1))

    np_vertices = np.ndarray([8,3], dtype=float)
    new_vertices = np.ndarray([8,3], dtype=float)

    sx=0.85
    sy=0.85
    sz_top=0.6
    sz_bottom = -0.0005
    np_vertices[0,0:3] = [sx,sy,sz_top] 
    np_vertices[1,0:3] = [sx,-sy,sz_top] 
    np_vertices[2,0:3] = [-sx,-sy,sz_top] 
    np_vertices[3,0:3] = [-sx,sy,sz_top] 
    np_vertices[4,0:3] = [sx,sy,sz_bottom] 
    np_vertices[5,0:3] = [sx,-sy,sz_bottom] 
    np_vertices[6,0:3] = [-sx,-sy,sz_bottom] 
    np_vertices[7,0:3] = [-sx,sy,sz_bottom] 

    vertices = o3d.utility.Vector3dVector(np_vertices)  

    bbox=o3d.geometry.AxisAlignedBoundingBox.create_from_points(vertices)

    pcd_cropped = original_pcd.crop(bbox)


    #Plane segmentation --------------------

    plane_model, inlier_idxs = pcd_cropped.segment_plane(distance_threshold=0.05,ransac_n=3, num_iterations=100)

    a, b, c, d = plane_model
    pcd_table = pcd_cropped.select_by_index(inlier_idxs, invert=False)
    pcd_table.paint_uniform_color([1, 0, 0])

    pcd_objects = pcd_cropped.select_by_index(inlier_idxs, invert=True)

    #clustering --------------------------

    labels = pcd_objects.cluster_dbscan(eps=0.12, min_points=70, print_progress=True)

    group_idxs = list(set(labels))
    group_idxs.remove(-1)  # remove last group because its the group on the unassigned points
    num_groups = len(group_idxs)
    #colormap = cm.Pastel1(range(0, num_groups))

    pcd_separate_objects = []
    for group_idx in group_idxs:  # Cycle all groups, i.e.,

        group_points_idxs = list(locate(labels, lambda x: x == group_idx))

        pcd_separate_object = pcd_objects.select_by_index(group_points_idxs, invert=False)

        #color = colormap[group_idx, 0:3]
        #pcd_separate_object.paint_uniform_color(color)
        pcd_separate_objects.append(pcd_separate_object)

        
    print("\n\nSeparated Objects:")
    print(len(pcd_separate_objects))
    print("\n\n")

    # Save each separated object as a PLY file
    output_folder = "./output_point_clouds"
    os.makedirs(output_folder, exist_ok=True)

    # Apaga (se tiver algum) os arquivos na pasta output_point_clouds
    for file in [f for f in os.listdir(output_folder) if f.endswith(".ply")]:
        file_path = os.path.join(output_folder, file)
        os.remove(file_path)


    for i, pcd_separate_object in enumerate(pcd_separate_objects):
        output_filename = os.path.join(output_folder, f"{file_prefix}_object_{i}.ply")
        o3d.io.write_point_cloud(output_filename, pcd_separate_object)
        # print(f"Object {i} saved as {output_filename}")

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

    
    lista_propriedades_objetos = []

    for ply_file in ply_files:
        ply_path = os.path.join(output_folder, ply_file)
        point_cloud = o3d.io.read_point_cloud(ply_path)

        # COR --------------------------------------------------------------------------------
        print(point_cloud.colors)
        print("\n\n")
        cores = np.asarray(point_cloud.colors)
        cores_hsv = []
        for pixel in cores:
            hsv = colorsys.rgb_to_hsv(pixel[0], pixel[1], pixel[2])
            cores_hsv.append([hsv[0], hsv[1], hsv[2]])
        
        cores_hsv_asarray = np. asarray(cores_hsv)
        cor_media_hsv = np.mean(cores_hsv, axis=0)
        cor_moda_hsv = stats.mode(cores_hsv_asarray.round(decimals=2, out=None), keepdims=True)
        cor_max_hsv = np.max(cores_hsv, axis=0)
        cor_min_hsv = np.min(cores_hsv, axis=0)
        
        pontos = np.asarray(point_cloud.points)
        minimos = np.min(pontos, axis=0)
        maximos = np.max(pontos, axis=0)
        
        largura = maximos[0] - minimos[0]
        altura = maximos[2] - minimos[2] 

        racio = altura/largura

        area = len(cores) + len(cores) * 50


        area = area * 0.8 # MEXER NESTE VALOR THRESHOLD

        w = math.ceil(math.sqrt(area/racio))
        h = math.ceil(area / w)
        
        lista_propriedades_objetos.append({"cor_media": np.round(cor_media_hsv*255), "cor_moda": np.round(cor_moda_hsv.mode*255), "h": h, "w": w })

        # Visualize the point cloud
        o3d.visualization.draw_geometries([point_cloud, frame_table, frame_world], window_name=f"Nuvem de Pontos para {file_prefix}", zoom=1,
                                           front=view['trajectory'][0]['front'],
                                          lookat=view['trajectory'][0]['lookat'],
                                         up=view['trajectory'][0]['up'])
    

    bgr_image = cv2.imread(filename_rgb_image)
    bgr_image_copy = bgr_image.copy()
    height = bgr_image.shape[0]
    width = bgr_image.shape[1]
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    print("\n\n\n\n")

    for obj in range(0, len(lista_propriedades_objetos)):
        print("\n ------------Propriedades do Objecto: -----------")
        print(lista_propriedades_objetos[obj])
        print("(cores em HSV)")


    # Specify the folder path
    folder_path = "./output_cropped_images"
    # Verifica se há arquivos na pasta output_point_clouds
    output_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # Apaga os arquivos na pasta output_point_clouds
    for file in output_files:
        file_path = os.path.join(folder_path, file)
        os.remove(file_path)

    for num_object in range(0, len(lista_propriedades_objetos)):
        h = lista_propriedades_objetos[num_object]["h"]
        w = lista_propriedades_objetos[num_object]["w"]

        H_obj = (lista_propriedades_objetos[num_object]["cor_media"][0] + lista_propriedades_objetos[num_object]["cor_moda"][0][0]) / 2
        S_obj = (lista_propriedades_objetos[num_object]["cor_media"][1] + lista_propriedades_objetos[num_object]["cor_moda"][0][1]) / 2
        V_obj = (lista_propriedades_objetos[num_object]["cor_media"][2] + lista_propriedades_objetos[num_object]["cor_moda"][0][2]) / 2
        
        temos_mascara = 0
        tol = 10

        for keys, values in hsv_image_boundaries.items():    
        
            if (values["H"][0] < H_obj < values["H"][1]+tol) and (values["S"][0]-tol < S_obj < values["S"][1]+tol) and (values["V"][0]-tol < V_obj < values["V"][1]+tol):
                temos_mascara = 1
            
            if temos_mascara == 1:
                minH = values["H"][0]
                maxH = values["H"][1]
                minS = values["S"][0]
                maxS = values["S"][1]
                minV = values["V"][0]
                maxV = values["V"][1]
                break
    
        mask = cv2.inRange(hsv_image, (minH, minS, minV), (maxH, maxS, maxV))
        cv2.imshow("mask", mask)

        rectangle_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(rectangle_mask, (60, 120), (570, 400), 255, -1)

        inside_rect = cv2.bitwise_and(mask, rectangle_mask)
        cv2.imshow("Bitwise (inside rectangle)", inside_rect)

        kernel = np.ones((3,3),np.uint8)
        erode = cv2.erode(inside_rect, kernel, iterations=1)
        cv2.imshow("Erode Mask", erode)
        
        kernel = np.ones((5,5),np.uint8)
        dilation = cv2.dilate(erode,kernel,iterations = 2)
        cv2.imshow("Dilated Mask", dilation)
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilation, connectivity=8)
        
        distancias = []
        area_ref = lista_propriedades_objetos[num_object]["w"] * lista_propriedades_objetos[num_object]["h"]
        print("area_ref" + str(area_ref))

        # Display the result    
        for i in range(1, num_labels):
            # Skip the background label (label 0)
            x, y, w, h, area = stats[i]
            print("area" + str(area))
            distancias.append(abs(area - (area_ref*0.6)))

        index_distancia_minima = np.argmin(distancias)
        x, y, w, h, area = stats[index_distancia_minima+1]

        w = lista_propriedades_objetos[num_object]["w"] *1.1
        h = lista_propriedades_objetos[num_object]["h"] * 1.3
        centro = centroids[index_distancia_minima+1]

        cropped_image = bgr_image_copy[round(centro[1])-round(h/2):round(centro[1])+round(h/2), round(centro[0])-round(w/2):round(centro[0])+round(w/2)]
    
        # Ensure the folder exists, create it if necessary
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        obj_prefix = f"{num_object:02}"
        filename_cropped_image = f'{file_prefix}_{obj_prefix}.png'

        # Construct the full path
        full_path = os.path.join(folder_path, filename_cropped_image)

        # Save the image to the specified folder
        cv2.imwrite(full_path, cropped_image)

        cv2.rectangle(bgr_image, (round(centro[0])-round(w/2), round(centro[1])-round(h/2)), (round(centro[0])+round(w/2), round(centro[1])+round(h/2)), (255, 0, 0), 2)

        cv2.imshow("Real Image", bgr_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()