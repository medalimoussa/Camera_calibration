import cv2
import numpy as np
import glob
import os

# ----------------- CONFIG -----------------
CHECKERBOARD = (8, 6)       # (columns, rows) = inner corners
SQUARE_SIZE = 1.0           # in your chosen unit (e.g., cm or meters)

IMAGES_PATH = "Calibration data/*.JPG"  # <-- update if needed
OUTPUT_YAML = "Camera_calibration.yaml"
RESIZE_VISUALIZATION = True
VIS_WIDTH = 800             # Preview window width for corner detection
# ------------------------------------------

# Prepare object points (0,0,0) ... (e.g., (0,0,0), (1,0,0), ...)
objp = np.zeros((CHECKERBOARD[1] * CHECKERBOARD[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D points in board coordinate
imgpoints = []  # 2D points in image plane

# Collect calibration images
images = sorted(glob.glob(IMAGES_PATH))
print(f"📷 Found {len(images)} calibration images.")

# Detect corners in each image
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # You can add flags for better detection under difficult lighting
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
           | cv2.CALIB_CB_NORMALIZE_IMAGE
           | cv2.CALIB_CB_FAST_CHECK)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)

    if ret:
        objpoints.append(objp)
        cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners)

        # Visualization of detected corners
        img_display = cv2.drawChessboardCorners(img.copy(), CHECKERBOARD, corners, ret)
        if RESIZE_VISUALIZATION:
            h, w = img_display.shape[:2]
            scale = VIS_WIDTH / w
            img_display = cv2.resize(img_display, (int(w * scale), int(h * scale)))
        cv2.imshow("Detected Corners", img_display)
        cv2.waitKey(100)
        print(f"✅ Processed {idx+1}/{len(images)}")
    else:
        print(f"❌ Failed to detect corners in: {fname}")

cv2.destroyAllWindows()

# --- Calibration ---
print("\n🔧 Running calibration...")
image_size = gray.shape[::-1]  # (width, height)
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, None, None
)

# Print calibration results
print("\n🎯 Calibration results:")
print("Camera Matrix (K):\n", K)
print("Distortion Coefficients:\n", dist.ravel())

# --- Reprojection Error Check ---
total_error = 0
errors = []
for i, (objp_i, imgp_i, rvec, tvec) in enumerate(zip(objpoints, imgpoints, rvecs, tvecs)):
    imgp_proj, _ = cv2.projectPoints(objp_i, rvec, tvec, K, dist)
    err = cv2.norm(imgp_i, imgp_proj, cv2.NORM_L2) / len(imgp_proj)
    errors.append(err)
    total_error += err
    print(f"Image {i+1}: Reprojection error = {err:.4f} pixels")
avg_error = total_error / len(objpoints)
print(f"\nAverage reprojection error: {avg_error:.4f} pixels")

# --- Undistortion Visualization ---
print("\n🖼️ Showing undistorted examples...")
for fname in images[:3]:  # show first three for brevity
    img = cv2.imread(fname)
    undistorted = cv2.undistort(img, K, dist)
    combined = np.hstack((img, undistorted))
    if RESIZE_VISUALIZATION:
        h, w = combined.shape[:2]
        scale = (2 * VIS_WIDTH) / w
        combined = cv2.resize(combined, (int(w * scale), int(h * scale)))
    cv2.imshow("Original | Undistorted", combined)
    cv2.waitKey(500)
cv2.destroyAllWindows()

# --- Save to ORB-SLAM2 YAML format ---
with open(OUTPUT_YAML, 'w') as f:
    f.write("%YAML:1.0\n\n")
    f.write("# Camera Parameters\n\n")
    f.write(f"Camera.fx: {K[0,0]}\n")
    f.write(f"Camera.fy: {K[1,1]}\n")
    f.write(f"Camera.cx: {K[0,2]}\n")
    f.write(f"Camera.cy: {K[1,2]}\n\n")
    f.write(f"Camera.k1: {dist[0,0]}\n")
    f.write(f"Camera.k2: {dist[0,1]}\n")
    f.write(f"Camera.p1: {dist[0,2]}\n")
    f.write(f"Camera.p2: {dist[0,3]}\n")
    f.write(f"Camera.k3: {dist[0,4]}\n\n")
    f.write("Camera.fps: 30.0\n")
    f.write("Camera.RGB: 1\n\n")
    f.write("# ORB Parameters\n")
    f.write("ORBextractor.nFeatures: 1000\n")
    f.write("ORBextractor.scaleFactor: 1.2\n")
    f.write("ORBextractor.nLevels: 8\n")
    f.write("ORBextractor.iniThFAST: 20\n")
    f.write("ORBextractor.minThFAST: 7\n\n")
    f.write("# Viewer Parameters\n")
    f.write("Viewer.KeyFrameSize: 0.05\n")
    f.write("Viewer.KeyFrameLineWidth: 1\n")
    f.write("Viewer.GraphLineWidth: 0.9\n")
    f.write("Viewer.PointSize: 2\n")
    f.write("Viewer.CameraSize: 0.08\n")
    f.write("Viewer.CameraLineWidth: 3\n")
    f.write("Viewer.ViewpointX: 0\n")
    f.write("Viewer.ViewpointY: -0.7\n")
    f.write("Viewer.ViewpointZ: -1.8\n")
    f.write("Viewer.ViewpointF: 500\n")

print(f"\n📁 ORB-SLAM2 config file saved to: {OUTPUT_YAML}")
