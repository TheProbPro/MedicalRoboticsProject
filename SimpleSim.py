import os
import numpy as np
import roboticstoolbox as rtb
import roboticstoolbox.backends.swift
from spatialmath import SE3, SO3
from spatialmath.base import *
import argparse
import copy
import sys
import logging
import time
import numpy as np
from pytransform3d import transformations as pt
# from paho.mqtt import client as mqtt_client

from roboticstoolbox.backends.swift import Swift
import spatialgeometry as sg


_logger = logging.getLogger('sdu_controllers')

if __name__ == '__main__':
    frequency = 500.0  # Hz
    dt = 1 / frequency

    env = Swift()

    robot = rtb.models.UR3()
    #robot = rtb.models.URDF.UR3()
    
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
    
    
    env.launch(realtime=True, rate=500)#frequency=500, realtime=True)
    env.add(box)
    env.add(robot)
    #env.add(plane_axes)
    #env.add(tip_axes)
    env.step(0.002)

    # get tip in base as position and quaternion
    T_base_tip_pq = pt.pq_from_transform(T_base_tip)
    T_base_tip_pos_init = T_base_tip_pq[0:3]
    T_base_tip_quat_init = T_base_tip_pq[3:7]

    # Set initial circle target
    counter = 0.0
    
    robot.q = q
    env.step(0.002)
    time.sleep(2)

    # print("BEFORE")
    # print("T_tcp_tip: ", T_tcp_tip)
    # print("T_base_tip: ", T_base_tip)
    # print("T_base_tcp: ", T_base_tcp)

    
    # The controller parameters can be changed, the following parameters corresponds to the default,
    # and we set them simply to demonstrate that they can be changed.
    M_init = np.diag([2.5, 2.5, 2.5])
    D_init = np.diag([500, 500, 1500])
    K_init = np.diag([54, 54, 0])

    Mo_init = np.diag([0.25, 0.25, 0.25])
    Do_init = np.diag([300, 300, 300])
    Ko_init = np.diag([0.5, 0.5, 0.5])


    f_target = np.array([0, 0, 25])

    # normal_estimator = SurfaceNormalEstimator(freq=500, gamma1=0, gamma2=1, beta=0.9, r=np.zeros((3, 1)), initial_n=[0, 0, 1], K_Ln=100)

    # Generate a time vector
    ts = np.arange(start=0, stop=30, step=dt)

    # Plane stiffness
    K_plane = 1e5

    
    
    for t in ts:
        # Calculate current plane rotation
        T_base_plane = SE3.Rt(SO3.Rx(np.pi/12 * np.sin(0.5*t)), [0.5, 0, 0.2])
        plane_axes.T = T_base_plane
        box.T = T_base_plane

        # get current robot pose and velocity
        T_base_tcp = robot.fkine(robot.q, end="tool0")
        T_base_tcp_pq = pt.pq_from_transform(T_base_tcp)
        p_current = T_base_tcp_pq[:3]
        #v = (p_current - p_prev)/dt
        
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

        # nc = (rotx(30, unit='deg') @ np.array([[0,0,1]]).T).flatten()
        # print("Surface normal: ", nc)
        # mqtt.publish(nc)

        # Align gain matrices to surface normal
        
        # the input position and orientation is given as tip in base
        
        # mqtt_force.publish(((R_new @ f_target)/nm_force))

        # get circle target
        # x_desired = get_circle_target(T_base_tip_pos_init, counter)
        # print("x_desired: ", x_desired)
        # print("x_desired_rot: ", x_desired_rot)

        #print("D = ", D)
        #print("M = ", M)
        #print("K = ", K)

        # TODO: Uncomment this and comment the next line to disable orientation adjustment based on normal estimate
        # adm_controller.set_desired_frame(x_desired, quaternion.from_float_array(T_base_tip_quat_init))

        # adm_controller.set_desired_frame(x_desired, q_normal)
        
        # rotate output from tip to TCP before sending it to the robot
        robot.q = q
        env.step(0.002)

        
    print("done")
