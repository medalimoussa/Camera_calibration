# 📷 Camera Calibration with OpenCV

This repository provides two Python scripts for calibrating a camera using a checkerboard pattern. 
It generates camera intrinsic parameters file (compatible with [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2)).

---

## 📁 Folder Structure

![image](https://github.com/user-attachments/assets/abe57d1f-1248-46b2-9aa3-070098833f35)


---

## 🛠️ Requirements

- Python 3.x
- OpenCV
- NumPy

Install dependencies:

```bash
pip install opencv-python numpy

🖼️ Preparing Calibration Images

    Place your checkerboard images in the Calibration data/ folder.

    Image format should be .JPG (or change the IMAGES_PATH pattern in the script).

    Default checkerboard size is 8×6 inner corners (i.e., 9×7 actual squares).

🚀 How to Use
🔹 Standard Calibration (full-resolution)

python calibrate.py

    Detects corners

    Calibrates the camera

    Saves results to Camera_calibration.yaml

    Shows undistorted sample images

🔹 Resized Calibration (faster)

python calibrate_resized.py

    Resizes images to 640×480 before processing

    Saves output to Camera_ORBSLAM2.yaml

⚙️ Configuration Options

Edit directly in the script as needed:

    CHECKERBOARD – Tuple indicating number of inner corners: (columns, rows)

    SQUARE_SIZE – Real-world size of one square (e.g., in cm or meters)

    TARGET_SIZE – Resize dimensions for faster calibration (calibrate_resized.py only)

✅ Output Sample

Terminal output:

📷 Found 20 calibration images.
✅ Processed 20/20
🎯 Calibration results:
Camera Matrix (K):
[[fx  0 cx]
 [0  fy cy]
 [0   0  1]]
Distortion Coefficients:
[k1 k2 p1 p2 k3]
Average reprojection error: 0.2517 px

YAML file snippet:

%YAML:1.0
Camera.fx: 1234.56
Camera.fy: 1234.56
Camera.cx: 640.0
Camera.cy: 360.0
Camera.k1: -0.123
Camera.k2: 0.456
...

🔍 Notes

    Use 10–20 images taken from different angles.

    Good lighting and full visibility of the checkerboard improve accuracy.

    Lower reprojection error = better calibration quality.
