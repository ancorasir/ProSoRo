# Modules

## Overview

This directory contains the modules that are used in the project. The structure of the directory is as follows:

```plaintext
modules/
├── README.md
├── __init__.py                         # Initialize the modules package
├── cam/
│   ├── __init__.py                     # Initialize the camera module
│   ├── config/
│   │   ├── cam_1280x720.yaml           # Camera matrix and distortion coefficients
│   │   └── detector_params.yaml        # ArUco detector parameters
│   ├── camera.py                       # Camera class
│   ├── real_camera.py                  # RealCamera class
│   └── virtual_camera.py               # VirtualCamera class
├── interface/
│   ├── __init__.py                     # Initialize the interface module
│   └── interface.py                    # Interface class
├── led/
│   ├── __init__.py                     # Initialize the LED module
│   ├── led_arduino.ino                 # Arduino code for LED control
│   └── led.py                          # LED class
└── protobuf/
    ├── generate_pb.sh                  # Script to generate Python files from .proto files
    ├── image_msg_pb2.py                # Generated Python file for image message
    ├── image_msg.proto                 # Image message definition
    ├── nodes_msg_pb2.py                # Generated Python file for nodes message
    ├── nodes_msg.proto                 # Nodes message definition
    ├── pose_msg_pb2.py                 # Generated Python file for pose message
    └── pose_msg.proto                  # Pose message definition
```

## Camera module

In the `cam/` directory, the camera module is defined. The `camera.py` is the base class for the camera, and the `real_camera.py` and `virtual_camera.py` are the derived classes for the real and virtual cameras, respectively. The `config` directory contains the camera matrix and distortion coefficients in `cam_{frame_width}x{frame_height}.yaml` and the ArUco detector parameters in `detector_params.yaml`.

### Camera class

In Camera class, there are several parameters:

- `mark_size`: The size of the ArUco marker in meters.
- `pose_accuracy`: The accuracy of the pose estimation in millimeters.
- `frame_width`: The width of the camera frame in pixels.
- `frame_height`: The height of the camera frame in pixels.
- `filter_on`: A flag to enable the Average Filter for pose estimation.
- `filter_frame`: The number of frames for the Average Filter.
- `fps`: The frame rate of the camera in frames per second.
- `exposure`: The exposure of the camera.

And there are several methods that can be used:

- `read_img()`: Read the camera frame.
- `read_pose_img()`: Read the camera frame and estimate the pose of the ArUco marker.
- `pose2ref()`: Convert the pose of the ArUco marker to the reference pose.
- `pose_as_euler()`: Convert the pose of the ArUco marker to Euler angles in radians.
- `pose_as_quat()`: Convert the pose of the ArUco marker to quaternions.
- `pose_as_matrix()`: Convert the pose of the ArUco marker to rotation matrix.
- `pose_filter()`: Apply the Average Filter to the pose estimation.
- `release()`: Release the camera.

Before initializing the camera, the camera matrix and distortion coefficients should be stored in the `cam_{frame_width}x{frame_height}.yaml` file. The camera matrix and distortion coefficients can be obtained by calibrating the camera.

### RealCamera class

In RealCamera class, the camera is initialized based on the previous Camera class, then images and poses can be read and published to defined address. Several parameters should be set:

- `pose_address`: The address to publish the pose.
- `image_address`: The address to publish the image.
- `frame_width`: The width of the camera frame in pixels.
- `frame_height`: The height of the camera frame in pixels.
- `mark_size`: The size of the ArUco marker in meters.
- `fps`: The frame rate of the camera in frames per second.
- `exposure`: The exposure of the camera.

After initializing the RealCamera, images and poses can be published continuously by calling the `start()` method.

### VirtualCamera class

In VirtualCamera class, images and poses is read from the folder that contains previously recorded images and poses. Several parameters should be set:

- `pose_address`: The address to publish the pose.
- `image_address`: The address to publish the image.
- `src_path`: The path to the folder that contains images and poses.

After initializing the VirtualCamera, images and poses can be published continuously by calling the `start()` method.

## Interface module

The interface is built with [Plotly Dash](https://plotly.com/dash/), visualizing the camera frame, the pose, force and shape of the ProSoRo. The `interface.py` defines the class for the interface.

## LED module

In the `led/` directory, the LED module is defined. The `led.py` is the class for the LED control, and the `led_arduino.ino` is the Arduino code for LED control.

### LED class

In LED class, there are several parameters:

- `pose_address`: The address to subscribe the pose.
- `ser_port`: The serial port of the Arduino.
- `ser_baud`: The baud rate of the serial port.

Poses are subscribed to calculate which LED corresponds to the pose direction and what light intensity should be set. The LED control is implemented by sending the LED index and light intensity to the Arduino through the serial port.

### LED Arduino code

In the `led_arduino.ino`, `FastLED` library is used to control the LED. The LED index and light intensity are received from the serial port, and the corresponding LED is set to the specified light intensity.

## Protobuf module

In the `protobuf/` directory, `.proto` files define the message structures of the image, nodes, and pose, and `.py` files are generated by the `generate_pb.sh` script. These generated Python files are used to publish and subscribe messages between the camera, interface, and LED modules.
