import os
import numpy as np
import roboticstoolbox.backends.swift
from quaternion import quaternion
from spatialmath import SE3, SO3
from spatialmath.base import *
import argparse
import copy
import sys
import logging
import time
import numpy as np
import quaternion
from pytransform3d import transformations as pt
from sdu_controllers import __version__
from sdu_controllers.controllers.admittance_controller_position import AdmittanceControllerPosition
from sdu_controllers.estimators.surface_normal_estimator import SurfaceNormalEstimator
# from paho.mqtt import client as mqtt_client

from roboticstoolbox.backends.swift import Swift
import spatialgeometry as sg
from UR5_URDF import UR5_URDF
from sdu_controllers.utils.math_util import null

# class Mqtt():
#     def __init__(self, topic = "inigo",client_id = 'Robot'):
#         #self.broker="broker.emqx.io"
#         self.broker = '0.0.0.0'
#         self.port = 1883
#         self.topic = topic
#         # Generate a Client ID with the publish prefix.
#         self.client_id = client_id
#
#         def on_connect(client, userdata, flags, rc):
#             if rc == 0:
#                 print("Connected to MQTT Broker!")
#             else:
#                 print("Failed to connect, return code %d\n", rc)
#
#         self.client = mqtt_client.Client(self.client_id)
#         # client.username_pw_set(username, password)
#         self.client.on_connect = on_connect
#         self.client.connect(self.broker, self.port)
#
#     def publish(self, value):
#         msg = f"{value[0]},{value[1]},{value[2]}"
#         result = self.client.publish(self.topic, payload=msg, qos=0, retain=False)
#         status = result[0]
#         if status == 0:
#             print(f"Send `{msg}` to topic `{self.topic}`")
#         else:
#             print(f"Failed to send message to topic {self.topic}")


_logger = logging.getLogger('sdu_controllers')


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Admittance controller (trajectory tracking example)")
    parser.add_argument(
        "--version",
        action="version",
        version="sdu_controllers {ver}".format(ver=__version__))
    parser.add_argument(
        dest="ip",
        help="IP address of the robot",
        type=str,
        metavar="<IP address of the robot>")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO)
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG)
    return parser.parse_args(args)


def get_circle_target(pose, timestep, radius=0.075, freq=0.5):
    circ_target = copy.deepcopy(pose)
    circ_target[0] = pose[0] + radius * np.cos((2 * np.pi * freq * timestep))
    circ_target[1] = pose[1] + radius * np.sin((2 * np.pi * freq * timestep))
    return circ_target


def get_circle_target_rotated(init_pos, R_current, timestep, radius=0.075, freq=0.5):
    x_circle = radius * np.cos((2 * np.pi * freq * timestep))
    y_circle = radius * np.sin((2 * np.pi * freq * timestep))
    p_circle = np.array([[x_circle, y_circle, 0]]).T
    print(R_current)
    print(p_circle)
    circ_target = init_pos + (R_current @ p_circle).flatten()

    return circ_target


def compute_alignment_to_normal(nc, R_base_tip):
    z_axis = nc
    y_axis_old = R_base_tip[:3, 1]
    x_axis_old = R_base_tip[:3, 0]
    x_axis_new = np.cross(y_axis_old, z_axis)
    # print("norm of new x: ", np.linalg.norm(x_axis_new))

    if not np.linalg.norm(x_axis_new) < 1e-5:
        print("Case 1")
        x_axis_new = x_axis_new / np.linalg.norm(x_axis_new)
        y_axis_new = np.cross(z_axis, x_axis_new)
        y_axis_new = y_axis_new / np.linalg.norm(y_axis_new)
    else:
        print("Case 2")
        y_axis_new = np.cross(z_axis, x_axis_old)
        # print("norm of new y: ", np.linalg.norm(y_axis_new))
        y_axis_new = y_axis_new / np.linalg.norm(y_axis_new)
        x_axis_new = np.cross(y_axis_new, z_axis)
        x_axis_new = x_axis_new / np.linalg.norm(x_axis_new)

    R_new = np.column_stack((x_axis_new, y_axis_new, z_axis))
    # print("R new: ", R_new)
    return R_new


def compute_gains_aligned_to_normal(nc, dict_gains):
    if nc.shape != (1, 3):
        nc_row = np.reshape(nc, (1, 3))
    else:
        nc_row = nc

    # Compute null space of nc
    null_nc = null(nc_row)
    M = np.zeros((3, 3))
    K = np.zeros((3, 3))
    D = np.zeros((3, 3))
    Mo = np.zeros((3, 3))
    Ko = np.zeros((3, 3))
    Do = np.zeros((3, 3))

    for i in range(0, 2):
        prim = null_nc[:, i:i + 1] @ null_nc[:, i:i + 1].T
        M = M + prim * dict_gains['M_free']
        K = K + prim * dict_gains['K_free']
        D = D + prim * dict_gains['D_free']
        Mo = Mo + prim * dict_gains['Mo_contact']
        Ko = Ko + prim * dict_gains['Ko_contact']
        Do = Do + prim * dict_gains['Do_contact']

    nc_col = np.reshape(nc, (3, 1))
    prim = nc_col @ nc_col.T
    M = M + prim * dict_gains['M_contact']
    K = K + prim * dict_gains['K_contact']
    D = D + prim * dict_gains['D_contact']
    Mo = Mo + prim * dict_gains['Mo_free']
    Ko = Ko + prim * dict_gains['Ko_free']
    Do = Do + prim * dict_gains['Do_free']
    return M, K, D, Mo, Ko, Do


if __name__ == '__main__':
    frequency = 500.0  # Hz
    dt = 1 / frequency
   
    # make mqtt object
    # mqtt = Mqtt()
    # mqtt_force = Mqtt("force", 'robot_force')


    dir_path = os.path.dirname(os.path.realpath(__file__))
    urdf_path = dir_path + "/ur5.urdf"
    env = Swift()

    robot = UR5_URDF(urdf_path)
    q1 = np.deg2rad([-150, 0, -90, 0, 90, 0])
    T_base_tcp = SE3.Rt(SO3.Ry(np.pi), np.array([0.4, 0.1, 0.4]))
    q, success, iter, searches, residual = robot.ik_LM(T_base_tcp, q0=q1, end='tool0')
    robot.q = q

    # define a tip wrt. the TCP frame
    T_tcp_tip = SE3(0, 0, 0.194)
    T_tip_tcp = T_tcp_tip.inv()
    T_base_tip = T_base_tcp * T_tcp_tip
    T_base_tip_init = SE3(T_base_tip)

    tip_axes = sg.Axes(length=0.1, pose=T_base_tip)
    T_base_plane = SE3.Rt(SO3.Rx(np.pi / 12 * np.sin(0 / 2)), [0.5, 0, 0.2])
    box = sg.Cuboid([1, 1, 0.001], pose=T_base_plane)
    plane_axes = sg.Axes(length=0.1, pose=T_base_plane)

    env.launch(frequency=500, realtime=True)
    env.add(robot)
    env.add(box)
    env.add(plane_axes)
    env.add(tip_axes)
    env.step(0.002)

    # get tip in base as position and quaternion
    T_base_tip_pq = pt.pq_from_transform(T_base_tip)
    T_base_tip_pos_init = T_base_tip_pq[0:3]
    T_base_tip_quat_init = T_base_tip_pq[3:7]

    # Set initial circle target
    counter = 0.0
    # x_desired = get_circle_target_rotated(T_base_tip, counter)
    x_desired = get_circle_target(T_base_tip_pos_init, counter)

    T_base_tip_circle = SE3(pt.transform_from_pq(np.hstack((x_desired, T_base_tip_quat_init))))
    T_base_tcp_circle = T_base_tip_circle @ T_tip_tcp
    q, success, iter, searches, residual = robot.ik_LM(T_base_tcp_circle, q0=robot.q, end='tool0')
    robot.q = q
    tip_axes.T = T_base_tip_circle
    env.step(0.002)
    time.sleep(2)

    # print("BEFORE")
    # print("T_tcp_tip: ", T_tcp_tip)
    # print("T_base_tip: ", T_base_tip)
    # print("T_base_tcp: ", T_base_tcp)

    adm_controller = AdmittanceControllerPosition(start_position=x_desired, start_orientation=T_base_tip_quat_init,
                                                  start_ft=np.array([0, 0, 0, 0, 0, 0]))

    # The controller parameters can be changed, the following parameters corresponds to the default,
    # and we set them simply to demonstrate that they can be changed.
    M_init = np.diag([2.5, 2.5, 2.5])
    D_init = np.diag([500, 500, 1500])
    K_init = np.diag([54, 54, 0])

    Mo_init = np.diag([0.25, 0.25, 0.25])
    Do_init = np.diag([300, 300, 300])
    Ko_init = np.diag([0.5, 0.5, 0.5])

    adm_controller.M = M_init
    adm_controller.D = D_init
    adm_controller.K = K_init
    adm_controller.Mo = Mo_init
    adm_controller.Do = Do_init
    adm_controller.Ko = Ko_init

    gain_parameters = {
        'K_free': 54,
        'D_free': 200,
        'M_free': 2.5,
        'K_contact': 0.0,
        'D_contact': 200.0,
        'M_contact': 2.5,
        'Ko_free': 0.5,
        'Do_free': 300.0,
        'Mo_free': 0.25,
        'Ko_contact': 0.5,
        'Do_contact': 300.0,
        'Mo_contact': 0.25
    }

    f_target = np.array([0, 0, 25])

    normal_estimator = SurfaceNormalEstimator(freq=500, gamma1=0.2, gamma2=0.8, beta=0.9, r=np.zeros((3, 1)), initial_n=[0, 0, 1], K_Ln=100)
    # normal_estimator = SurfaceNormalEstimator(freq=500, gamma1=0, gamma2=1, beta=0.9, r=np.zeros((3, 1)), initial_n=[0, 0, 1], K_Ln=100)

    # Generate a time vector
    ts = np.arange(start=0, stop=30, step=dt)

    # Plane stiffness
    K_plane = 1e5

    T_base_tip_pq = pt.pq_from_transform(T_base_tcp_circle)
    p_prev = T_base_tip_pq[:3]
    q_prev = quaternion.from_float_array(T_base_tip_pq[3:])

    T_base_tcp_pq = pt.pq_from_transform(T_base_tcp_circle)
    q_old = quaternion.from_float_array(T_base_tcp_pq[3:])

    for t in ts:
        # Calculate current plane rotation
        T_base_plane = SE3.Rt(SO3.Rx(np.pi/12 * np.sin(0.5*t)), [0.5, 0, 0.2])
        plane_axes.T = T_base_plane
        box.T = T_base_plane

        # get current robot pose and velocity
        T_base_tcp = robot.fkine(robot.q, end="tool0")
        T_base_tcp_pq = pt.pq_from_transform(T_base_tcp)
        p_current = T_base_tcp_pq[:3]
        q_current = quaternion.from_float_array(T_base_tcp_pq[3:])
        v = (p_current - p_prev)/dt
        omega = (2*np.log(q_current*q_prev.conjugate())).imag/dt
        vel = np.array([*v, *omega])

        # print("AFTER")
        # print("T_tcp_tip: ", T_tcp_tip)
        # print("T_base_tip: ", T_base_tip)
        # print("T_base_tcp: ", T_base_tcp)
        T_base_tip = T_base_tcp @ T_tcp_tip
        tip_axes.T = T_base_tip

        # Calculate penetration distance into plane (z-axis projection)
        T_plane_tip = T_base_plane.inv() @ T_base_tip
        T_plane_tcp = T_base_plane.inv() @ T_base_tcp
        # print("T_plane_tip: ", T_base_tip)
        # print("T_plane_tcp: ", T_base_tcp)

        penetration = T_plane_tip.t[2]
        # print("penetration: ", penetration)

        # Calculate contact force
        if penetration < 0:
            F = -K_plane * penetration
            ft_plane = np.array([0.0, 0.0, F, 0.0, 0.0, 0.0])
        else:
            ft_plane = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # print("Force before noise: ", ft_plane)
        # ft_plane += np.array([*np.random.normal(0,1,3), *np.random.normal(0,0.01,3)])
        # ft_plane += np.array([*np.random.normal(0,0.0001,3), *np.random.normal(0,0.0000000001,3)])
        # print("Force after noise: ", ft_plane)

        f_base_tip = T_base_plane.R @ ft_plane[:3]
        mu_base_tip = T_base_plane.R @ ft_plane[3:]
        mu_tip = T_base_tip.inv().R @ mu_base_tip
        ft_base = np.array([*f_base_tip, *mu_base_tip])

        # get tip in base as position and quaternion
        T_base_tip_pq = pt.pq_from_transform(T_base_tip)
        T_base_tip_pos = T_base_tip_pq[0:3]
        T_base_tip_quat = T_base_tip_pq[3:7]

        normal_estimator.update(T_base_tcp.R, vel, ft_base)
        nc = normal_estimator.get_estimate().flatten()
        # nc = (rotx(30, unit='deg') @ np.array([[0,0,1]]).T).flatten()
        # print("Surface normal: ", nc)
        # mqtt.publish(nc)

        # Align gain matrices to surface normal
        if nc.shape != (1, 3):
            nc_row = np.reshape(nc, (1, 3))
        else:
            nc_row = nc

        # Calculate new orientation based on surface normal estimate
        R_new = compute_alignment_to_normal(-nc, T_base_tip_init.R)
        # print("R alignment = ", R_new)
        # print("original force: ", f_target)
        # print("rotated force: ", R_new @ f_target)
        q_normal = quaternion.from_rotation_matrix(R_new)

        # the input position and orientation is given as tip in base
        adm_controller.pos_input = T_base_tip_pos
        adm_controller.rot_input = quaternion.from_float_array(T_base_tip_quat)
        adm_controller.q_input = robot.q
        adm_controller.ft_input = np.hstack((f_base_tip + R_new @ f_target, mu_tip))
        nm_force = np.linalg.norm(R_new @ f_target)
        
        # mqtt_force.publish(((R_new @ f_target)/nm_force))

        # get circle target
        # x_desired = get_circle_target(T_base_tip_pos_init, counter)
        # print("x_desired: ", x_desired)
        x_desired_rot = get_circle_target_rotated(T_base_tip_pos_init, R_new, counter)
        # print("x_desired_rot: ", x_desired_rot)

        M, K, D, Mo, Ko, Do = compute_gains_aligned_to_normal(nc, gain_parameters)
        R_gains = np.linalg.inv(D) @ D_init
        # print("R gains = ", R_gains)
        adm_controller.M = M
        adm_controller.D = D
        adm_controller.K = K
        adm_controller.Mo = Mo
        adm_controller.Do = Do
        adm_controller.Ko = Ko

        #print("D = ", D)
        #print("M = ", M)
        #print("K = ", K)

        # TODO: Uncomment this and comment the next line to disable orientation adjustment based on normal estimate
        # adm_controller.set_desired_frame(x_desired, quaternion.from_float_array(T_base_tip_quat_init))

        # adm_controller.set_desired_frame(x_desired, q_normal)
        adm_controller.set_desired_frame(x_desired_rot, q_normal)

        # step the execution of the admittance controller
        adm_controller.step()
        output = adm_controller.get_output()
        output_position = output[0:3]
        output_quat = output[3:7]

        # rotate output from tip to TCP before sending it to the robot
        T_base_tip_out = SE3(pt.transform_from_pq(np.hstack((output_position, output_quat))))
        T_base_tcp_out = T_base_tip_out @ T_tip_tcp

        q, success, iter, searches, residual = robot.ik_LM(T_base_tcp_out, q0=robot.q, end='tool0')
        robot.q = q
        env.step(0.002)

        T_base_tcp_out_pq = pt.pq_from_transform(T_base_tcp_out)
        p_prev = T_base_tcp_out_pq[:3]
        q_prev = quaternion.from_float_array(T_base_tcp_out_pq[3:])
        counter += dt
        q_old = q_normal

    print("done")

