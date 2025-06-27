import cv2
import numpy as np
import glob
import os

# ----------------- CONFIG -----------------
CHECKERBOARD = (8, 6)       # (columns, rows) = inner corners
SQUARE_SIZE = 1.0           # in your chosen unit (e.g., cm or meters)

IMAGES_PATH = "Calibration data/*.JPG"  # <-- update if needed
OUTPUT_YAML = "Camera_ORBSLAM2.yaml"
RESIZE_VISUALIZATION = True
VIS_WIDTH = 800             # Preview window width
TARGET_SIZE = (640, 480)    # Resize input images for faster processing
# ------------------------------------------

# Prepare object points (0,0,0)...
objp = np.zeros((CHECKERBOARD[1] * CHECKERBOARD[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D points in board coords
imgpoints = []  # 2D points in image plane

# Gather image file names
images = sorted(glob.glob(IMAGES_PATH))
print(f"📷 Found {len(images)} calibration images.")

# Detect chessboard corners
for idx, fname in enumerate(images):
    img_full = cv2.imread(fname)
    img = cv2.resize(img_full, TARGET_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

        # Visualization: draw and optionally resize
        vis = cv2.drawChessboardCorners(img.copy(), CHECKERBOARD, corners, ret)
        if RESIZE_VISUALIZATION:
            h, w = vis.shape[:2]
            scale = VIS_WIDTH / w
            vis = cv2.resize(vis, (int(w * scale), int(h * scale)))
        cv2.imshow("Detected Corners", vis)
        cv2.waitKey(100)
        print(f"✅ Processed {idx+1}/{len(images)}")
    else:
        print(f"❌ Failed to detect corners in: {fname}")

cv2.destroyAllWindows()

# --- Calibration ---
print("\n🔧 Running calibration...")
# Use resized image size for calibration
image_size = TARGET_SIZE
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, None, None
)

print("\n🎯 Calibration results:")
print("Camera Matrix (K):\n", K)
print("Distortion Coefficients:\n", dist.ravel())

# --- Reprojection Error Check ---
total_error = 0
for i, (op, ip, rv, tv) in enumerate(zip(objpoints, imgpoints, rvecs, tvecs)):
    proj, _ = cv2.projectPoints(op, rv, tv, K, dist)
    err = cv2.norm(ip, proj, cv2.NORM_L2) / len(proj)
    total_error += err
    print(f"Image {i+1}: reprojection error = {err:.4f} px")
avg_error = total_error / len(objpoints)
print(f"\nAverage reprojection error: {avg_error:.4f} px")

# --- Undistortion Visualization ---
print("\n🖼️ Showing undistorted examples...")
newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, image_size, alpha=0, newImgSize=image_size)
map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, image_size, cv2.CV_16SC2)

for fname in images[:3]:
    img_full = cv2.imread(fname)
    img_rs = cv2.resize(img_full, TARGET_SIZE)
    und = cv2.remap(img_rs, map1, map2, interpolation=cv2.INTER_LINEAR)
    x, y, w, h = roi
    if h > 0 and w > 0:
        und = und[y:y+h, x:x+w]
    # side-by-side
    h0, w0 = img_rs.shape[:2]
    und_s = cv2.resize(und, (w0, h0))
    combo = np.hstack((img_rs, und_s))
    if RESIZE_VISUALIZATION:
        h1, w1 = combo.shape[:2]
        scale = (2 * VIS_WIDTH) / w1
        combo = cv2.resize(combo, (int(w1 * scale), int(h1 * scale)))
    cv2.imshow("Original | Undistorted", combo)
    cv2.waitKey(0)
cv2.destroyAllWindows()

# --- Save to ORB-SLAM2 YAML format ---
with open(OUTPUT_YAML, 'w') as f:
    f.write("%YAML:1.0\n\n# Camera Parameters\n\n")
    f.write(f"Camera.fx: {K[0,0]}\nCamera.fy: {K[1,1]}\nCamera.cx: {K[0,2]}\nCamera.cy: {K[1,2]}\n\n")
    f.write(f"Camera.k1: {dist[0,0]}\nCamera.k2: {dist[0,1]}\nCamera.p1: {dist[0,2]}\nCamera.p2: {dist[0,3]}\nCamera.k3: {dist[0,4]}\n\n")
    f.write("Camera.fps: 30.0\nCamera.RGB: 1\n\n# ORB Parameters\nORBextractor.nFeatures: 1000\nORBextractor.scaleFactor: 1.2\nORBextractor.nLevels: 8\nORBextractor.iniThFAST: 20\nORBextractor.minThFAST: 7\n\n# Viewer Parameters\nViewer.KeyFrameSize: 0.05\nViewer.KeyFrameLineWidth: 1\nViewer.GraphLineWidth: 0.9\nViewer.PointSize: 2\nViewer.CameraSize: 0.08\nViewer.CameraLineWidth: 3\nViewer.ViewpointX: 0\nViewer.ViewpointY: -0.7\nViewer.ViewpointZ: -1.8\nViewer.ViewpointF: 500\n")
print(f"\n📁 ORB-SLAM2 config file saved to: {OUTPUT_YAML}")
