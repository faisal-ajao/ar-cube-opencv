# Camera Calibration and Augmented Reality Cube (OpenCV)

## Overview

A computer vision pipeline for camera calibration and real-time augmented reality.  
The system estimates camera parameters using a chessboard pattern and overlays a 3D cube onto a live video feed.

Chessboard calibration patterns can be downloaded from the official OpenCV website.  
Example (9×6 pattern):  
https://github.com/opencv/opencv/blob/master/doc/pattern.png

------------------------------------------------------------------------

## Demo

![AR Cube Demo](assets/demo.gif)

------------------------------------------------------------------------

## Features

- Chessboard-based camera calibration  
- Real-time pose estimation (SolvePnP)  
- 3D cube projection onto live camera feed  
- Rotation and floating animation  
- Live undistortion preview  

------------------------------------------------------------------------

## Project Structure

    ar-cube-calibration/
    ├── camera_calibration.py
    ├── ar_cube.py
    ├── utils.py
    ├── calibration.xml
    ├── chessboard_images/
    ├── assets/
    │   └── demo.gif
    ├── README.md
    └── requirements.txt

------------------------------------------------------------------------

## How It Works

1. Capture chessboard images  
2. Calibrate camera (intrinsics + distortion)  
3. Detect chessboard in live video  
4. Estimate pose using SolvePnP  
5. Project and render 3D cube  

------------------------------------------------------------------------

## Installation

```bash
git clone https://github.com/faisal-ajao/ar-cube-opencv.git
cd ar-cube-opencv
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Usage

```bash
python camera_calibration.py
python ar_cube.py
```

------------------------------------------------------------------------

## Requirements

- Python 3.8+  
- OpenCV  
- NumPy  

------------------------------------------------------------------------

## License

[MIT License](https://opensource.org/licenses/MIT)

------------------------------------------------------------------------

## References

- OpenCV Camera Calibration  
  https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html  

- OpenCV SolvePnP (Pose Estimation)  
  https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html  

- Chessboard Calibration Pattern  
  https://github.com/opencv/opencv/blob/master/doc/pattern.png

------------------------------------------------------------------------

## Author

Ajao Faisal
