#!/usr/bin/env python

import os
import sys
import zmq
import base64
import torch
import yaml
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table

sys.path.append(os.path.split(sys.path[0])[0])
from protobuf import image_msg_pb2, pose_msg_pb2

sys.path.append(os.path.split(os.path.split(sys.path[0])[0])[0])
from models.mvae import MVAE


# Set the pose subscriber
class PoseSubscriber:
    def __init__(self, address: str):
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

    def receive_message(self):
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


class ImageSubscriber:
    def __init__(self, address: str):
        """Subscribe the image message from the camera.

        This class is used to subscribe the image message from the camera.
        The image message is received from the camera and set to the image.
        The image keeps updating.

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

        # Set the image to zero
        self.image = np.zeros((360, 640, 3))

    def receive_message(self):
        """Receive the image message from the camera.

        In the function, the image message is received from the camera.
        The image message is parsed and set to the image.
        The image is returned.

        Args:
            None

        Returns:
            image: np.ndarray, the image message from the camera
        """

        # Parse the image message
        img = image_msg_pb2.Image()
        img.ParseFromString(self.subscriber.recv())

        # Set the image from the message
        self.image = np.frombuffer(img.data, np.uint8)

        # Return the image
        return self.image


# Mesh filter
def mesh_filter(nodes: np.ndarray, nodes_list: list, filter_frame: int):
    """Filter the mesh data.

    This function is used to filter the mesh data by the number of frames.
    The mesh data is appended to the list if the length of the list is less than the number of frames.
    Otherwise, the first element is popped from the list and the mesh data is appended to the list.
    The mean of the list is calculated and returned as the filtered mesh data.

    Args:
        nodes: np.ndarray, the mesh data to be filtered
        nodes_list: list, the list of mesh data
        filter_frame: int, the number of frames to be filtered

    Returns:
        nodes: np.ndarray, the filtered mesh data
        nodes_list: list, the updated list of mesh data
    """

    # Check the length of the list
    if len(nodes_list) < filter_frame:
        # Append the nodes to the list
        nodes_list.append(nodes)
    else:
        # Pop the first element from the list
        nodes_list.pop(0)
        nodes_list.append(nodes)

    # Calculate the mean of the list
    nodes = np.mean(np.array(nodes_list), axis=0)

    # Return the nodes and nodes list
    return nodes, nodes_list


class Interface:
    def __init__(
        self, cam_num: str = "cam_0", filter_on: bool = True, filter_frame: int = 5
    ) -> None:
        """Initialize the interface.

        This function is used to initialize the interface.
        The pose and image are subscribed from the camera.
        The model is loaded and the mesh is filtered.

        Args:
            None

        Returns:
            None
        """

        # Set the pose and image address
        with open("config/address.yaml", "r") as f:
            pub_sub_adress_dict = yaml.load(f.read(), Loader=yaml.Loader)

        # Create a pose and image subscriber
        self.pose_subscriber = PoseSubscriber(
            address=pub_sub_adress_dict[cam_num]["pose"]
        )
        self.image_subscriber = ImageSubscriber(
            address=pub_sub_adress_dict[cam_num]["image"]
        )

        # Load model
        print("Loading model...")

        # module_type
        module_type_list = [
            "cylinder",
        ]
        # loss = alpha / (1 + alpha) * recon_loss + 1/ (1 + alpha) * pred_loss + zeta * z_loss + gamma * kl_loss
        # recon_pred_scale: coresponds to alpha in the paper
        # recon_coeff: recon_pred_scale / (1 + recon_pred_scale)
        # pred_coeff: 1 / (1 + recon_pred_scale)
        recon_pred_scale = 1
        # z_coeff: coresponds to zeta in the paper
        z_coeff = 1
        # kl_coeff: coresponds to gamma in the paper
        kl_coeff = 0.1
        # x_dim_list: the dimension of the input data
        x_dim_dict = {
            "cylinder": [6, 6, 2736],
        }
        # h1_dim_list: the dimension of the hidden layer 1
        h1_dim_list = [16, 16, 1024]
        # h2_dim_list: the dimentsion of the hidden layer 2
        h2_dim_list = [32, 32, 256]
        # z_dim: the dimension of the latent space
        z_dim = 32

        # Load model and normalization parameters
        self.model_dict = {}
        self.mu_dict = {}
        self.std_dict = {}
        self.surf_node_dict = {}
        self.surf_triangles_dict = {}

        for module_type in module_type_list:
            pth_path = (
                "models/pths/mvae_"
                + module_type
                + "_"
                + str(recon_pred_scale)
                + "_"
                + str(z_coeff)
                + "_"
                + str(kl_coeff)
                + "_"
                + str(z_dim)
                + ".pth"
            )
            model = MVAE(
                x_dim_list=x_dim_dict[module_type],
                h1_dim_list=h1_dim_list,
                h2_dim_list=h2_dim_list,
                z_dim=z_dim,
                recon_pred_scale=recon_pred_scale,
                z_coeff=z_coeff,
                kl_coeff=kl_coeff,
            )
            model.load_state_dict(torch.load(pth_path))
            model.eval()
            self.model_dict[module_type] = model

            # load normalization parameters
            data_path = "data/" + module_type + "/"
            mu = np.load(data_path + "mu.npy")
            std = np.load(data_path + "std.npy")
            self.mu_dict[module_type] = mu
            self.std_dict[module_type] = std

            # original coordinates
            template_path = "templates/" + module_type + "/"
            surf_node = np.loadtxt(template_path + "surf_node.txt", delimiter=",")
            self.surf_node_dict[module_type] = surf_node

            # surface triangles
            surf_triangles = np.loadtxt(
                template_path + "surf_triangles.txt", delimiter=","
            ).astype(int)
            self.surf_triangles_dict[module_type] = surf_triangles

        # Initialize model with module type
        module_type = "cylinder"
        model = self.model_dict[module_type]
        mu = self.mu_dict[module_type]
        std = self.std_dict[module_type]
        surf_node = self.surf_node_dict[module_type]
        surf_triangles = self.surf_triangles_dict[module_type]

        # Initialize pose
        pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Convert numpy array to tensor
        pose_tensor = torch.from_numpy((pose - mu[0:6]) / std[0:6]).float()
        # Predict
        force_pred = model.forward_with_index(pose_tensor, 0, 1).detach().numpy()
        node_pred = model.forward_with_index(pose_tensor, 0, 2).detach().numpy()
        # Denormalize data
        force_pred = force_pred * std[6:12] + mu[6:12]
        node_pred = (node_pred * std[12:] + mu[12:]).reshape(-1, 3)

        # Get node
        node_curr = surf_node.copy()
        node_curr[: node_pred.shape[0], 1:] += node_pred

        # Filter
        self.filter_on = filter_on
        self.filter_frame = filter_frame
        self.nodes_pred_list = []

        # Create mesh figure
        mesh_fig = go.Mesh3d(
            x=-node_curr[:, 1],
            y=node_curr[:, 2],
            z=node_curr[:, 3],
            i=surf_triangles[:, 0],
            j=surf_triangles[:, 1],
            k=surf_triangles[:, 2],
            colorscale="Viridis",
            autocolorscale=False,
            colorbar=dict(
                title=dict(
                    text="Displacement (mm)",
                    side="right",
                    font=dict(
                        size=16,
                        family="Arial",
                    ),
                ),
                tickvals=[0, 2, 4, 6],
                tickfont=dict(
                    size=14,
                    family="Arial",
                ),
                len=0.5,
            ),
            cmin=0,
            cmax=6,
            intensity=np.linalg.norm((node_curr - surf_node)[:, 1:], axis=1),
            opacity=1,
            showscale=True,
        )
        print("Model loaded!")

        # Load camera
        print("Loading camera...")
        pose = self.pose_subscriber.receive_message()
        buf = self.image_subscriber.receive_message()
        pose[:] = [pose[0], pose[1], -pose[2], -pose[3], -pose[4], -pose[5]]
        print("Camera loaded!")

        # Convert image to base64
        img_date_url = "data:image/jpeg;base64," + str(base64.b64encode(buf))[2:-1]

        # Create Dash app
        self.app = Dash(__name__)
        self.app.layout = html.Div(
            children=[
                html.Div(
                    children="Digital module",
                    style={
                        "fontSize": 36,
                        "fontFamily": "Arial",
                        "margin": "10px",
                    },
                ),
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        html.Img(
                                            id="img",
                                            src=img_date_url,
                                            style={
                                                "width": "100%",
                                                "height": "100%",
                                            },
                                        ),
                                    ],
                                    style={
                                        "width": "100%",
                                        "height": "100%",
                                        "margin": "5px",
                                    },
                                ),
                                html.Div(
                                    children=[
                                        dash_table.DataTable(
                                            id="table_pose",
                                            columns=[
                                                {
                                                    "name": [
                                                        "ArUco Displacement",
                                                        "Dx (mm)",
                                                    ],
                                                    "id": "Dx",
                                                },
                                                {
                                                    "name": [
                                                        "ArUco Displacement",
                                                        "Dy (mm)",
                                                    ],
                                                    "id": "Dy",
                                                },
                                                {
                                                    "name": [
                                                        "ArUco Displacement",
                                                        "Dz (mm)",
                                                    ],
                                                    "id": "Dz",
                                                },
                                                {
                                                    "name": [
                                                        "ArUco Rotation",
                                                        "Rx (rad)",
                                                    ],
                                                    "id": "Rx",
                                                },
                                                {
                                                    "name": [
                                                        "ArUco Rotation",
                                                        "Ry (rad)",
                                                    ],
                                                    "id": "Ry",
                                                },
                                                {
                                                    "name": [
                                                        "ArUco Rotation",
                                                        "Rz (rad)",
                                                    ],
                                                    "id": "Rz",
                                                },
                                            ],
                                            merge_duplicate_headers=True,
                                            style_table={"overflowX": "auto"},
                                            style_cell={
                                                "height": "auto",
                                                "minWidth": "50px",
                                                "width": "50px",
                                                "maxWidth": "50px",
                                                "whiteSpace": "normal",
                                                "textAlign": "center",
                                                "fontSize": "14px",
                                                "fontFamily": "Arial",
                                            },
                                        ),
                                    ],
                                    style={
                                        "width": "100%",
                                        "height": "100%",
                                        "margin": "5px",
                                    },
                                ),
                                html.Div(
                                    children=[
                                        dash_table.DataTable(
                                            id="table_force",
                                            columns=[
                                                {
                                                    "name": [
                                                        "Predicted Force",
                                                        "Fx (N)",
                                                    ],
                                                    "id": "Fx",
                                                },
                                                {
                                                    "name": [
                                                        "Predicted Force",
                                                        "Fy (N)",
                                                    ],
                                                    "id": "Fy",
                                                },
                                                {
                                                    "name": [
                                                        "Predicted Force",
                                                        "Fz (N)",
                                                    ],
                                                    "id": "Fz",
                                                },
                                                {
                                                    "name": [
                                                        "Predicted Torque",
                                                        "Tx (Nmm)",
                                                    ],
                                                    "id": "Tx",
                                                },
                                                {
                                                    "name": [
                                                        "Predicted Torque",
                                                        "Ty (Nmm)",
                                                    ],
                                                    "id": "Ty",
                                                },
                                                {
                                                    "name": [
                                                        "Predicted Torque",
                                                        "Tz (Nmm)",
                                                    ],
                                                    "id": "Tz",
                                                },
                                            ],
                                            merge_duplicate_headers=True,
                                            style_table={"overflowX": "auto"},
                                            style_cell={
                                                "height": "auto",
                                                "minWidth": "50px",
                                                "width": "50px",
                                                "maxWidth": "50px",
                                                "whiteSpace": "normal",
                                                "textAlign": "center",
                                                "fontSize": "14px",
                                                "fontFamily": "Arial",
                                            },
                                        ),
                                    ],
                                    style={
                                        "width": "100%",
                                        "height": "100%",
                                        "margin": "5px",
                                    },
                                ),
                            ],
                            style={
                                "display": "inline-block",
                                "width": "40%",
                                "height": "100%",
                                "margin": "5px",
                            },
                        ),
                        html.Div(
                            children=[
                                dcc.RadioItems(
                                    id="module_type",
                                    options=[
                                        {
                                            "label": "Cylinder",
                                            "value": "cylinder",
                                        },
                                    ],
                                    value="cylinder",
                                    style={
                                        "width": "100%",
                                        "fontSize": "14px",
                                        "fontFamily": "Arial",
                                        "margin": "5px",
                                        "textAlign": "center",
                                    },
                                    inline=True,
                                ),
                                dcc.Graph(
                                    id="mesh",
                                    figure=go.Figure(
                                        data=[mesh_fig],
                                    ),
                                    style={"width": "100%", "height": "100%"},
                                ),
                            ],
                            style={
                                "display": "inline-block",
                                "width": "60%",
                                "height": "100%",
                                "margin": "5px",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "height": "100%",
                        "margin": "5px",
                    },
                ),
                dcc.Interval(
                    id="interval",
                    interval=33,
                ),
            ],
            style={
                "margin": "5px",
            },
        )

        # Update mesh
        @self.app.callback(
            [
                Output("mesh", "figure"),
                Output("table_pose", "data"),
                Output("table_force", "data"),
                Output("img", "src"),
            ],
            [
                Input("interval", "n_intervals"),
                Input("module_type", "value"),
            ],
        )
        def update(n_intervals, module_type):
            """Update the figure.

            Args:
                n_intervals: int, the number of intervals
                module_type: str, the type of the module

            Returns:
                mesh_fig: go.Figure, the mesh figure
                pose: dict, the pose data
                force: dict, the force data
                img_date_url: str, the image data url
            """

            # Load model with module type
            model = self.model_dict[module_type]
            mu = self.mu_dict[module_type]
            std = self.std_dict[module_type]
            surf_node = self.surf_node_dict[module_type]
            surf_triangles = self.surf_triangles_dict[module_type]

            # Get pose and image from camera
            pose = self.pose_subscriber.receive_message()
            buf = self.image_subscriber.receive_message()
            pose[:] = [pose[0], pose[1], -pose[2], -pose[3], -pose[4], -pose[5]]
            # Convert pose to tensor
            pose_tensor = torch.from_numpy((pose - mu[0:6]) / std[0:6]).float()
            # Predict
            force_pred = model.forward_with_index(pose_tensor, 0, 1).detach().numpy()
            node_pred = model.forward_with_index(pose_tensor, 0, 2).detach().numpy()
            # Denormalize data
            force_pred = force_pred * std[6:12] + mu[6:12]
            node_pred = (node_pred * std[12:] + mu[12:]).reshape(-1, 3)

            # Filter
            if self.filter_on:
                node_pred, nodes_pred_list = mesh_filter(
                    node_pred, self.nodes_pred_list, self.filter_frame
                )

            # Get node
            node_curr = surf_node.copy()
            node_curr[: node_pred.shape[0], 1:] += node_pred

            # Create mesh figure
            mesh_fig = go.Mesh3d(
                x=-node_curr[:, 1],
                y=node_curr[:, 2],
                z=node_curr[:, 3],
                i=surf_triangles[:, 0],
                j=surf_triangles[:, 1],
                k=surf_triangles[:, 2],
                colorscale="Viridis",
                autocolorscale=False,
                colorbar=dict(
                    title=dict(
                        text="Displacement (mm)",
                        side="right",
                        font=dict(
                            size=16,
                            family="Arial",
                        ),
                    ),
                    tickvals=[0, 2, 4, 6],
                    tickfont=dict(
                        size=14,
                        family="Arial",
                    ),
                    len=0.5,
                ),
                cmin=0,
                cmax=6,
                intensity=np.linalg.norm((node_curr - surf_node)[:, 1:], axis=1),
                lighting=dict(
                    ambient=1,
                    specular=0,
                    diffuse=0,
                ),
                opacity=1,
                showscale=True,
            )

            # Create image data url
            img_date_url = "data:image/jpeg;base64," + str(base64.b64encode(buf))[2:-1]

            # Return mesh figure, pose data, force data, and image data url
            return (
                {
                    "data": [mesh_fig],
                    "layout": go.Layout(
                        scene=dict(
                            camera=dict(
                                eye=dict(
                                    x=1,
                                    y=-1,
                                    z=1,
                                ),
                                projection=dict(
                                    type="orthographic",
                                ),
                            ),
                            xaxis=dict(
                                nticks=4,
                                range=[-50, 50],
                            ),
                            yaxis=dict(
                                nticks=4,
                                range=[-50, 50],
                            ),
                            zaxis=dict(
                                nticks=4,
                                range=[0, 60],
                            ),
                            aspectmode="manual",
                            aspectratio=go.layout.scene.Aspectratio(
                                x=1,
                                y=1,
                                z=0.75,
                            ),
                        ),
                        margin=dict(
                            l=0,
                            r=0,
                            b=0,
                            t=0,
                            pad=0,
                        ),
                    ),
                },
                [
                    {
                        "Dx": ["{:.3f}".format(pose[0])],
                        "Dy": ["{:.3f}".format(pose[1])],
                        "Dz": ["{:.3f}".format(pose[2])],
                        "Rx": ["{:.3f}".format(pose[3])],
                        "Ry": ["{:.3f}".format(pose[4])],
                        "Rz": ["{:.3f}".format(pose[5])],
                    }
                ],
                [
                    {
                        "Fx": ["{:.3f}".format(force_pred[0])],
                        "Fy": ["{:.3f}".format(force_pred[1])],
                        "Fz": ["{:.3f}".format(force_pred[2])],
                        "Tx": ["{:.3f}".format(force_pred[3])],
                        "Ty": ["{:.3f}".format(force_pred[4])],
                        "Tz": ["{:.3f}".format(force_pred[5])],
                    }
                ],
                img_date_url,
            )

    def run(self):
        """Run the loop.

        In the function, the pose and image are received from the camera.
        The mesh is updated and displayed.

        Args:
            None

        Returns:
            None
        """

        # Run the app
        self.app.run(debug=False)


if __name__ == "__main__":
    # Initialize the interface
    interface = Interface(cam_num="cam_0", filter_on=True, filter_frame=5)
    # Run the loop
    interface.run()
