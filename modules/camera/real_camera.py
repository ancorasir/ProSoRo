#!/usr/bin/env python

import os
import sys
import time
import zmq
import cv2
import yaml
import numpy as np

sys.path.append(os.path.split(sys.path[0])[0])
from protobuf import image_msg_pb2, pose_msg_pb2
from camera import Camera


class PosePublisher:
    def __init__(self, address: str) -> None:
        """Publish the pose message.

        This class is used to publish the pose message.
        The pose message is from the camera.

        Args:
            address: str, the address of the pose

        Returns:
            None
        """

        # Create a publisher to publish messages
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind(address)

        # Set the pose message
        self.pose = pose_msg_pb2.Pose()

    def publish_message(self, pose: np.ndarray) -> None:
        """Publish the pose message.

        In the function, the pose is turned into the pose message and published.

        Args:
            pose: np.ndarray, the pose message to be published

        Returns:
            None
        """

        # Set the pose message
        self.pose.position[:] = pose[:3].tolist()
        self.pose.rotation[:] = pose[3:].tolist()

        # Publish the pose message
        self.publisher.send(self.pose.SerializeToString())


# Set the image publisher
class ImagePublisher:
    def __init__(
        self, address: str, frame_width: int = 640, frame_height: int = 360
    ) -> None:
        """Publish the image message.

        This class is used to publish the image message.
        The image message is from the camera and includes the arrays, width, and height.

        Args:
            address: str, the address of the camera
            frame_width: int, the width of the image
            frame_height: int, the height of the image

        Returns:
            None
        """

        # Create a publisher to publish messages
        self.context = zmq.Context()
        self.publisher = self.context.socket(zmq.PUB)
        self.publisher.bind(address)

        # Set the image message
        self.image = image_msg_pb2.Image()
        self.image.width = frame_width
        self.image.height = frame_height

    def publish_message(self, img: np.ndarray) -> None:
        """Publish the image message.

        In the function, the image is turned into the image message and published.

        Args:
            img: np.ndarray, the image message to be published

        Returns:
            None
        """

        # Encode the image
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 50])

        # Set the image message
        self.image.data = buf.tobytes()

        # Publish the image message
        self.publisher.send(self.image.SerializeToString())


# Set the camera
class RealCamera:
    def __init__(
        self,
        pose_address: str,
        image_address: str,
        frame_width: int = 640,
        frame_height: int = 360,
        mark_size: float = 0.012,
        fps: int = 30,
        exposure: int = -3,
    ) -> None:
        """Publish the pose and image message.

        This function is used to keep publishing the pose and image message.
        The pose and image message are from the camera.
        PosePublisher and ImagePublisher are used to publish the pose and image message.

        Args:
            pose_address: str, the address of the camera
            image_address: str, the address of the camera
            cam_num: int, the camera number
            frame_width: int, the width of the image
            frame_height: int, the height of the image
            fps: int, the frame per second
            exposure: int, the exposure of the camera

        Returns:
            None
        """

        # Create a pose publisher
        self.pose_publisher = PosePublisher(address=pose_address)
        # Create an image publisher
        self.image_publisher = ImagePublisher(
            address=image_address, frame_width=frame_width, frame_height=frame_height
        )

        # Create a camera
        self.camera = Camera(
            mark_size=mark_size,
            pose_accuracy=[2, 2, 2, 2, 2, 2],
            frame_width=frame_width,
            frame_height=frame_height,
            filter_on=False,
            filter_frame=5,
            fps=fps,
            exposure=exposure,
        )

        self.fps = 0

    def start(self):

        # Start the camera
        start_time = time.time()
        frame_count = 0
        while True:
            # Get pose and image from camera
            pose, img = self.camera.read_pose_img()
            # Convert the pose to the reference pose
            pose = self.camera.pose2ref(pose)
            # Convert the pose to the euler pose
            pose = self.camera.pose_as_euler(pose)
            # Publish the pose and image
            self.pose_publisher.publish_message(pose)
            self.image_publisher.publish_message(img)

            # Calculate the fps
            frame_count += 1
            if frame_count == 50:
                self.fps = frame_count / (time.time() - start_time)
                start_time = time.time()
                frame_count = 0


if __name__ == "__main__":
    # Set the pose and image address
    with open("config/address.yaml", "r") as f:
        pub_sub_adress_dict = yaml.load(f.read(), Loader=yaml.Loader)
    pose_address = pub_sub_adress_dict["cam_0"]["pose"]
    image_address = pub_sub_adress_dict["cam_0"]["image"]
    # Set the frame width and height
    frame_width = 1280
    frame_height = 720
    # Set the mark size
    mark_size = 0.008
    # Set the fps
    fps = 30
    # Set the exposure
    exposure = -4

    # Run the main loop
    camera = RealCamera(
        pose_address=pose_address,
        image_address=image_address,
        frame_width=frame_width,
        frame_height=frame_height,
        mark_size=mark_size,
        fps=fps,
        exposure=exposure,
    )
    camera.start()
