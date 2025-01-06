import os
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from kornia.feature import LoFTR
import open3d as o3d

# Ensures ELAS library can be imported
elas_path = os.path.join(os.path.dirname(__file__), 'ELAS')
sys.path.append(elas_path)

from ELAS.elas import ELAS  # ELAS: Efficient Large-scale Stereo Matching from of JBF-Stereo Copyright (c) 2023, Eijiro Shibusawa
from ELAS.elas_params import elas_params  # Parameters for the ELAS algorithm

def LoftrKeypointsAndMatching(left_image: np.ndarray, right_image: np.ndarray) -> tuple:
    """
    Extract and match keypoints between stereo images using LOFTR
    LoFTR is a deep learning-based feature matching method that eliminates the need for traditional
    descriptors like SIFT or ORB or Sobel Operations.

    :param left_image: Grayscale left stereo image.
    :param right_image: Grayscale right stereo image.
    :return: (keypoints_left, keypoints_right, matches)
             - keypoints_left: Keypoints in the left image.
             - keypoints_right: Corresponding keypoints in the right image.
             - matches: Index mapping of matched keypoints.
    """
    matcher = LoFTR(pretrained="outdoor").cuda()  # LOFTR with pre-trained outdoor weight
    matcher.eval()

    # normalizing image to range [0,1] and converting to PyTorch tensors
    left_tensor = torch.tensor(left_image / 255.0).unsqueeze(0).unsqueeze(0).float().cuda()
    right_tensor = torch.tensor(right_image / 255.0).unsqueeze(0).unsqueeze(0).float().cuda()

    # Feature matching without gradient computation
    with torch.no_grad():
        input_dict = {"image0": left_tensor, "image1": right_tensor}
        correspondences = matcher(input_dict)

    # Extracting matched keypoints from the correspondences dictionary
    keypoints_left = correspondences["keypoints0"].cpu().numpy()
    keypoints_right = correspondences["keypoints1"].cpu().numpy()

    # LoFTR uses 1-to-1 keypoint pairing, so simple index-based match array
    matches = np.arange(len(keypoints_left))

    return keypoints_left, keypoints_right, matches

def estimateEgoMotion(matches: np.ndarray, keypoints1: np.ndarray, keypoints2: np.ndarray, intrinsic_matrix: np.ndarray) -> tuple:
    """
    Estimate ego-motion 'Relative Pose' between t wo images using matched keypoints
    Ego motion stands for relative movement of between two frames

    :param matches: array of indices mapping matched keypoint
    :param keypoints1: keypoints in first image
    :param keypoints2: keypoints in second
    :param intrinsic_matrix: intrinsic camera matrix
    :return: rotation matrix R and translation vector t
    """
    points1 = keypoints1[matches]
    points2 = keypoints2[matches]

    # Essential matrix calculation with using to handle outliers
    E, mask = cv2.findEssentialMat(points1, points2, intrinsic_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # Recovering pose from essential matrix
    _, R, t, _ = cv2.recoverPose(E, points1, points2, intrinsic_matrix)

    return R, t

def rectifyImages ( left_image, right_image, intrinsic_matrix, R, t ):
    """
    Rectify stereo images using rotation matrix and translation vector
    Simplifies disparity calculation by aligning epipolar lines are horizonal which is crucial for disparity map calculation
    :param left_image: left stereo image
    :param right_image: right stereo image
    :param intrinsic_matrix: intrinsic camera matrix
    :param R: rotation matrix
    :param t: translation vector
    :return: rectified left and right images, Q matrix
    """
    h, w = left_image.shape # dimensions of image

    # Stereo rectification on both images and reprojection matrix Q
    R1, R2, P1, P2,  Q, _, _ = cv2.stereoRectify(intrinsic_matrix, None, intrinsic_matrix, None, (w, h), R, t)

    #initliazing map for  rectification transformation
    map1x, map1y = cv2.initUndistortRectifyMap(intrinsic_matrix, None, R1, P1, (w, h), cv2.CV_32F)
    map2x, map2y = cv2.initUndistortRectifyMap(intrinsic_matrix, None, R2, P2, (w, h), cv2.CV_32F)

    # rectificaiton to input images
    left_rectified = cv2.remap(left_image, map1x, map1y, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_image, map2x, map2y, cv2.INTER_LINEAR)

    return left_rectified, right_rectified, Q

def computeDisparityELAS(left_image: np.ndarray, right_image: np.ndarray, min_disparity: int = 0, max_disparity: int = 128) -> tuple:
    """
    Compute the disparity map between rectified or normal (depends on choice) stereo images using ELAS
    
    ELAS is a stereo matching algorithm that computes disparity maps from stereo image pairs by Andreas Geiger et al.
    :param left_image: left stereo image
    :param right_image: right stereo image
    :param min_disparity: minimum disparity value
    :param max_disparity: maximum disparity value
    :return: normalized disparity map, raw disparity map
    """
    # Initialize ELAS parameters
    params = elas_params()
    params.disp_min = min_disparity
    params.disp_max = max_disparity
    params.add_corners = True

    # Initialize ELAS object
    elas_instance = ELAS(params)
    elas_instance.process(left_image, right_image)

    # Retrieve the left-to-right disparity map. The second return is the right-to-left disparity map (not used here).
    disparity_ref, _ = elas_instance.get_disparity()

    # Normalize the disparity map for visualization purposes (scale to 0–255).
    disparity_map_normalized = cv2.normalize(disparity_ref, None, 0, 255, cv2.NORM_MINMAX)
    disparity_map_normalized = np.uint8(disparity_map_normalized)

    return disparity_map_normalized, disparity_ref

def reconstruct3D(disparity_map: np.ndarray, Q: np.ndarray, left_image: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Reconstruct 3D point cloud from disparity map and reprojecti,on matrix
    :param disparity_map: disparity map comptued from stereo matching and elas
    :param Q: reprojection matrix
    :param left_rectified: rectified left image for coloring the point cloud (optional)
    return: open3d point cloud object
    """

    #mask to filter valid disparity values, ignore invalid or out-of-range disparities
    mask = (disparity_map > 0) & (disparity_map < 255)

    #reproject disparity to 3D space using reprojection matrix
    points_3d = cv2.reprojectImageTo3D(disparity_map, Q)
    points = points_3d[mask] # select only valid points

    #assign colors to 3d points based on the grayscale intensities of left image
    colors = cv2.cvtColor(left_image, cv2.COLOR_GRAY2RGB)[mask]/255.0 #normalize colors to [0,1]

    # create an open3d point cloud object and assign points and colours
    point_cloud = o3d.geometry.PointCloud()
    points -= points.mean(axis=0)
    scale_factor = np.max(np.linalg.norm(points, axis=1))
    points /= scale_factor

    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud



#Visualization Functions

def visualizeNormalizedPointCloud(point_cloud):
    """
    Visualize a normalized 3D point cloud with dynamic configuration.
    """
    points = np.asarray(point_cloud.points)
    points -= points.mean(axis=0)
    points /= np.max(np.linalg.norm(points, axis=1))
    point_cloud.points = o3d.utility.Vector3dVector(points)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)
    setView(vis, point_cloud)
    vis.run()
    vis.destroy_window()

def setView( vis , point_cloud):
    """
    Set the view of the open3d visualizer for better visualization
    initial pov can be adjusted based on the point cloud
    """
    view_control = vis.get_view_control()
    bounding_box = point_cloud.get_axis_aligned_bounding_box()
    center = bounding_box.get_center()
    extent = bounding_box.get_extent()

    max_extent = max(extent)
    camera_distance = max_extent * 2
    camera_position = center + np.array([0, 0, camera_distance])

    view_control.set_lookat(center)
    view_control.set_front([0, 0, -1])
    view_control.set_up([0, -1, 0])
    view_control.set_zoom(0.7)

def visualizeStereoResults(left_rect: np.ndarray, right_rect: np.ndarray, disparity_map: np.ndarray) -> None:
    """
    MatPlotLib based visualization of stereo rectified images and disparity map
    Visualize rectified images and the disparity map.
    :param left_rect: rectified left image
    :param right_rect: rectified right image
    :param disparity_map: disparity map
    :return None (displays the results)
    """
    """
    Visualize rectified images and the disparity map.
    """
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(left_rect, cmap='gray')
    plt.title("Rectified Left")
    plt.subplot(1, 2, 2)
    plt.imshow(right_rect, cmap='gray')
    plt.title("Rectified Right")
    plt.pause(0.001)  # Allow GUI to update
    plt.ioff() 
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.imshow(disparity_map, cmap='jet')
    plt.title("Disparity (ELAS)")
    plt.colorbar()
    plt.axis("off")
    plt.show()

def visualizePointCloud(point_cloud) :
    """
    Visualize 3D point cloud using Open3D
    :param point_cloud: Open3D point cloud object
    :return: None
    """
    points =  np.asarray(point_cloud.points)
    points -= points.mean(axis=0)
    points /= np.max(np.linalg.norm(points, axis=1))
    point_cloud.points = o3d.utility.Vector3dVector(points) 

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(point_cloud)
    setView(vis, point_cloud)
    vis.run()
    vis.destroy_window()

def readCalibrationFile(calibration_file: str) -> dict:
    """
    Parse calibration file and extract parameters.
    :param calibration_file: path to calibration file
    :return: dictionary of calibration parameters
    """
    calib_params = {}
    with open(calibration_file, 'r') as file:
        for line in file:
            if '=' in line:
                key, value = line.strip().split('=')
                if ';' in value:
                    calib_params[key] = np.array([list(map(float, row.split())) 
                                                  for row in value.strip('[]').split(';')])
                else:
                    calib_params[key] = float(value) if '.' in value else int(value)
    f = calib_params["cam0"][0, 0]
    cx0 = calib_params["cam0"][0, 2]
    cy0 = calib_params["cam0"][1, 2]
    cx1 = calib_params["cam1"][0, 2]
    baseline = calib_params["baseline"] / 1000.0

    Q = np.array([
        [1, 0, 0, -cx0],
        [0, 1, 0, -cy0],
        [0, 0, 0, f],
        [0, 0, -1 / baseline, (cx0 - cx1) / baseline]
    ])
    print("Recalculated Q Matrix:\n", Q)
    return calib_params