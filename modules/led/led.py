#!/usr/bin/env python

import os
import sys
import zmq
import serial
import yaml
import numpy as np

sys.path.append(os.path.split(sys.path[0])[0])
from protobuf import pose_msg_pb2


# Set the pose subscriber
class PoseSubscriber:
    def __init__(self, address: str) -> None:
        """Subscribe the pose message from the camera.

        This class is used to subscribe the pose message from the camera.
        The pose message is received from the camera and set to the pose.
        The pose keeps updating.

        Args:
            address: str, the address of the camera

        Returns:
            None
        """

        # Create a subscriber to subscribe messages
        self.context = zmq.Context()
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.connect(address)
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

        # Set the pose to zero
        self.pose = np.array([0, 0, 0, 0, 0, 0])

    def receive_message(self) -> np.ndarray:
        """Receive the pose message from the camera.

        In the function, the pose message is received from the camera.
        The pose message is parsed and set to the pose.
        The pose is returned.

        Args:
            None

        Returns:
            pose: np.ndarray, the pose message from the camera
        """

        # Parse the pose message
        pose = pose_msg_pb2.Pose()
        pose.ParseFromString(self.subscriber.recv())

        # Set the pose from the message
        self.pose = np.array([pose.position, pose.rotation]).flatten()

        # Return the pose
        return self.pose


class Led:
    def __init__(self, pose_address: str, ser_port: str, ser_baudrate: int) -> None:
        """Control the led light with the pose.

        This function is used to control the led light with the pose.
        The pose is received from the camera and used to control the led light.
        The led light position and strength are sent to the Arduino.

        Args:
            pose_address: str, the address of the camera
        """

        # Create a pose subscriber
        self.pose_subscriber = PoseSubscriber(address=pose_address)
        # Set the light strength
        self.light_strength = 10
        # Set the serial port
        self.ser = serial.Serial(port=ser_port, baudrate=ser_baudrate, timeout=1)

    def start(self) -> None:
        """Run the loop.

        In the function, the pose is received from the camera.
        The led position and light strength are calculated.

        Args:
            None

        Returns:
            None
        """

        while True:
            # Get pose from camera
            pose = self.pose_subscriber.receive_message()
            pose = [pose[0], -pose[1]]

            # Calculate the angle and distance of the pose
            pose_norm = np.linalg.norm(pose)
            pose_angle = np.arctan2(pose[1], pose[0])
            pose_angle = 270 - np.degrees(pose_angle)

            # Normalize the angle
            if pose_angle < 0:
                pose_angle += 360
            if pose_angle >= 360:
                pose_angle -= 360
            led_position = int(pose_angle / 30)
            if led_position == 12:
                led_position = 0

            # Calculate the light strength
            led_light = self.light_strength * pose_norm

            # Send the led position and light strength to the arduino
            led_str = str(led_position) + "," + str(int(led_light))
            self.ser.write(bytes(led_str, "utf-8"))


if __name__ == "__main__":
    # Set the pose address
    with open("config/address.yaml", "r") as f:
        adress_dict = yaml.load(f.read(), Loader=yaml.Loader)
    pose_address = adress_dict["cam_0"]["pose"]
    # Set Arduino serial port
    ser_port = adress_dict["led_0"]["port"]
    ser_baudrate = 115200

    # Run the main loop
    led = Led(pose_address=pose_address, ser_port=ser_port, ser_baudrate=ser_baudrate)
    led.start()
