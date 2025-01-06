import time 
import cv2
from reconstruction import (
    readCalibrationFile,
    LoftrKeypointsAndMatching,
    estimateEgoMotion,
    rectifyImages,
    computeDisparityELAS,
    reconstruct3D,
    visualizePointCloud,
    visualizeStereoResults,
)

from ELAS.elas import ELAS
from ELAS.elas_params import elas_params
from ELAS import elas

def main():
    """
    Main function to initialize the stereo pipeline:
    -Load Images
    -Read Calibration Parameters
    -Perform LoFTR Matching
    -Estimate Ego-Motion
    -Rectify Images
    -ELAS Disparity Calculation
    -Reconstruct 3D
    -Visualize Results
    """
    # Stereo Image Loading 
    left_image_path = 'test_images/moto_left.png'
    right_image_path = 'test_images/moto_right.png'
    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    if left_image is None:
        print(f"Error: Could not load images from {left_image_path}")
        return
    elif right_image is None:
        print(f"Error: Could not load images from {right_image_path}")
        return
    
    # Calibration Parameters Reading
    calibration_file = 'test_images/calib/calibMoto.txt'
    calibration_parameters = readCalibrationFile(calibration_file)
    intrinsic_matrix_left = calibration_parameters["cam0"]
    intrinsic_matrix_right = calibration_parameters["cam1"]
    baseline = calibration_parameters["baseline"] / 1000.0

    #Debugging For Calibration Parameters
    #print("\nCalibration Parameters:")
    #print(f"  Left Intrinsic:\n{intrinsic_matrix_left}")
    #print(f"  Right Intrinsic:\n{intrinsic_matrix_right}")
    #print(f"  Baseline: {baseline}")

    # LoFTR Feature Matching and Extraction
    start_time = time.time()
    keypoints_left, keypoints_right, loftr_matches = LoftrKeypointsAndMatching(left_image, right_image)
    match_time = time.time() - start_time

    #Debugging For LoFTR Results
    #print("\nLoFTR Results:")
    #print(f"  Total Matches Found: {len(loftr_matches)}")
    #print(f"  Matching Time for LoFTR:       {match_time:.2f} seconds")

    # Ego-Motion Estimation
    R, t = estimateEgoMotion(loftr_matches, keypoints_left, keypoints_right, intrinsic_matrix_left)
    
    #Debugging For Ego-Motion Estimation
    #print("\nEstimated Ego-motion:")
    #print(f"Rotation Matrix: {R}")
    #print(f"Translation Matrix: {t}")

    # Rectification of Images
    left_rectified, right_rectified, Q = rectifyImages(left_image, right_image, intrinsic_matrix_left, R, t)

    # ELAS Disparity Calculation
    disparity_left_normalized, disparity_left_raw = computeDisparityELAS(left_image, right_image)

    # ELAS Disparity Calculation Debugging
    #print("\nDisparity Calculation:")
    #print(f"Disparity Shape: {disparity_left_raw.shape}")

    # 3D Reconstruction
    point_cloud = reconstruct3D(disparity_left_raw, Q, left_rectified)

    # Visualizing 3D Point Cloud
    #print("\nVisualizing 3D Point Cloud...")
    visualizePointCloud(point_cloud)    

    # Visualizing Stereo Results
    visualizeStereoResults(left_rectified, right_rectified, disparity_left_normalized)

    print("Construction Completed")



if __name__ == "__main__":
    main()




    