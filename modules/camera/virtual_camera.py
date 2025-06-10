#!/usr/bin/env python

import os
import sys
import time
import zmq
import cv2
import yaml
import numpy as np
from scipy.signal import savgol_filter

sys.path.append(os.path.split(sys.path[0])[0])
from protobuf import image_msg_pb2, pose_msg_pb2


class PosePublisher:
    def __init__(self, address: str):
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

    def publish_message(self, pose: np.ndarray):
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
    def __init__(self, address: str, frame_width: int = 640, frame_height: int = 360):
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

    def publish_message(self, img: np.ndarray):
        """Publish the image message.

        In the function, the image is turned into the image message and published.

        Args:
            img: np.ndarray, the image message to be published

        Returns:
            None
        """

        # Encode the image
        _, buf = cv2.imencode(".jpeg", img, [cv2.IMWRITE_JPEG_QUALITY, 50])

        # Set the image message
        self.image.data = buf.tobytes()

        # Publish the image message
        self.publisher.send(self.image.SerializeToString())


class VirtualCamera:
    def __init__(
        self,
        pose_address: str,
        image_address: str,
        src_path: str,
        frame_width: int = 640,
        frame_height: int = 360,
        fps: int = 30,
    ):
        """Publish the pose and image message.

        This function is used to keep publishing the pose and image message.
        The pose and image message are from the recorded data.
        PosePublisher and ImagePublisher are used to publish the pose and image message.

        Args:
            pose_address: str, the address of the camera
            image_address: str, the address of the camera
            src_path: str, the path of the source data
            frame_width: int, the width of the image
            frame_height: int, the height of the image

        Returns:
            None
        """

        # Create a pose publisher
        self.pose_publisher = PosePublisher(address=pose_address)
        # Create an image publisher
        self.image_publisher = ImagePublisher(
            address=image_address, frame_width=frame_width, frame_height=frame_height
        )

        # Load the image
        self.img_list = []
        self.img_file_list = os.listdir(src_path + "img/")
        self.img_file_list.sort(key=lambda x: int(x.split(".")[0]))
        for img_file in self.img_file_list:
            img = cv2.imread(src_path + "img/" + img_file)
            self.img_list.append(img)
        print("Image loaded.")
        print("Length of the image list:", len(self.img_list))

        # Load the pose
        self.pose_list = np.loadtxt(src_path + "pose.txt", delimiter=",")
        # Filter the pose
        self.pose_list = savgol_filter(
            self.pose_list, window_length=10, polyorder=3, axis=0
        )
        self.frame_time = 1 / fps
        self.true_fps = 30

    def start(self):
        """Run the camera.

        This function is used to keep publishing the pose and image message.

        Args:
            None

        Returns:
            None
        """

        # Start the loop
        start_time = time.time()
        previous_time = start_time
        frame_count = 0
        while True:
            # Publish the pose and image
            self.pose_publisher.publish_message(self.pose_list[frame_count])
            self.image_publisher.publish_message(self.img_list[frame_count])

            frame_count += 1

            print("FPS:%.3f" % self.true_fps, end="\r")
            if frame_count % 50 == 0:
                # Calculate the fps
                self.true_fps = frame_count / (time.time() - start_time)

            # Make fps stable
            if time.time() - previous_time < self.frame_time:
                time.sleep(self.frame_time - (time.time() - previous_time))
                previous_time = time.time()

            if frame_count == len(self.img_list):
                frame_count = 0
                start_time = time.time()


if __name__ == "__main__":
    # Set the pose and image address
    with open("config/address.yaml", "r") as f:
        pub_sub_adress_dict = yaml.load(f.read(), Loader=yaml.Loader)
    pose_address = pub_sub_adress_dict["cam_0"]["pose"]
    image_address = pub_sub_adress_dict["cam_0"]["image"]
    # Set the source path
    src_path = "data/cylinder/recording/"

    # Run the camera
    camera = VirtualCamera(
        pose_address=pose_address,
        image_address=image_address,
        src_path=src_path,
        fps=30,
    )
    camera.start()
