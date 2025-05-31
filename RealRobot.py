import roboticstoolbox as rtb
import numpy as np
from spatialmath import SE3, SO3
from spatialmath.base import *
import logging
import time
from roboticstoolbox.backends.swift import Swift
import spatialgeometry as sg
from pytransform3d import transformations as pt
import quaternion

# UR-RTDE imports
import rtde_control, rtde_receive

# Admittance controller imports
from sdu_controllers import AdmittanceControllerPosition

# Import sesor module
from Sensors.UDP_client import UDPSensor

# Import BO libraries
from BaysianOptimization.Optimizer import BayesianOptimizer3D
from BaysianOptimization.Target import target_function2

_logger = logging.getLogger('Palpation_robot')

# TODO: Fix the instant flip in the second part of the trajectory, insert the trajectory validation, fix the BO, fix the remaining x points to sample

def point_along_line_3d(start, end, t, dt):
    """
    Calculates the 3D point on a line between `start` and `end` at time `t`, 
    moving with a frequency (interval) of `dt`.

    Args:
        start (tuple/list): Starting coordinate (x, y, z)
        end (tuple/list): Ending coordinate (x, y, z)
        t (float): Current time
        dt (float): Time step (frequency)

    Returns:
        tuple: Interpolated 3D point (x, y, z)
    """

    # Calculate the normalized parameter along the line
    alpha = t / dt

    # Clamp alpha to [0, 1] if you want to stay within the segment
    alpha = max(0.0, min(1.0, alpha))

    x = start[0] + alpha * (end[0] - start[0])
    y = start[1] + alpha * (end[1] - start[1])
    z = start[2] + alpha * (end[2] - start[2])

    return (x, y, z)

def xyToRobotPose(xy):
    x, y = xy
    z = -0.1
    robot_pose = SE3(x, y, z) * SE3(SO3.Rx(np.pi))# * robot.base
    return robot_pose

def xyzToRobotPose(xyz):
    x, y, z = xyz
    robot_pose = SE3(x, y, z) * SE3(SO3.Rx(np.pi))
    return robot_pose

def xyzrxryrzToSE3(robotpose):
    """
    Converts a robot pose given as [x, y, z, rx, ry, rz] into an SE3 object.
    The rotations are assumed to be in radians and follow the order of rotation around the x, y, and z axes.
    """
    x, y, z, rx, ry, rz = robotpose
    return SE3(x, y, z) * SE3(SO3.Rx(rx)) * SE3(SO3.Ry(ry)) * SE3(SO3.Rz(rz))

if __name__ == '__main__':
    # Initialize robot parameters
    freq = 125
    dt = 1 / freq
    max_force = 4.0 #N

    # Initialize the RTDE connection to the robot
    robot_ip = "192.168.1.105"
    rtde_robot = rtde_control.RTDEControlInterface(robot_ip)
    rtde_rec = rtde_receive.RTDEReceiveInterface(robot_ip)
    tool_offset = SE3(0, -0.075, 0.03) * SE3(SO3.Rx(np.pi/2))
    # Set initial pose
    #pose_init = [113.41, -70.30, 72.04, -1.13, 90.45, 180.51]
    #pose_init = [1.97972992, -1.22609448, 1.25759769, -0.01972218, 1.57842348, 3.15076803]
    pose_init = [(113/180)*np.pi, -np.pi/2, np.pi/2, 0, np.pi/2, 0]
    rtde_robot.moveJ(q=pose_init, speed=0.5, acceleration=0.1)

    # Initilize the force sensor
    sensor = UDPSensor()
    #sensor.start(unbias_data=False)
    sensor.start(unbias_data=True)


    # Initialize the admittance controller
    adm_controller = AdmittanceControllerPosition(freq)
    adm_controller.set_mass_matrix_position(np.identity(3) * 22.5)
    adm_controller.set_stiffness_matrix_position(np.identity(3) * 54)
    adm_controller.set_damping_matrix_position(np.identity(3) * 160)

    adm_controller.set_mass_matrix_orientation(np.identity(3) * 0.25)
    adm_controller.set_stiffness_matrix_orientation(np.identity(3) * 100)
    adm_controller.set_damping_matrix_orientation(np.identity(3) * 100)
    
    #Define frames
    #T_base_tcp = SE3(rtde_robot.getForwardKinematics(pose_init)[0:3])
    #    robot.fkine(robot.q))
    # T_tcp_tip = robot.tool
    # T_tip_tcp = T_tcp_tip.inv()
    # T_base_tip = T_base_tcp * T_tip_tcp
    # T_base_tip_init = SE3(T_base_tip)


    # Create the floor and box planes
    # floor_plane = sg.Cuboid([10, 10, 0.01], pose=robot.base, collision=True)
    box_plane = SE3(0.15, -0.4, 0)
    box_size = np.array([0.15, 0.15, 0.03])
    # box = sg.Cuboid(box_size, pose=box_plane, collision=True)

    time.sleep(1)

    # Initialize the optimizer
    optimizer = BayesianOptimizer3D(box_size, box_plane.t, grid_size=50)
    # Generate the first set of random samples
    samples = optimizer.init_random_samples()

    # Get the z-coordinate of the robot base
    ground_z = 0

    f = np.zeros(3)
    mu = np.zeros(3)
    
    force = 0
    contactPos = None

    # Run for x iterations
    for sample in samples:
        # Convert 2D to Robot coordinate2
        target_pose = xyToRobotPose(sample)
        begin_pose = target_pose
        target_flange_pose = target_pose * tool_offset.inv()

        # Compute the trajectory from the start configuration to the target configuration
        StartPose = rtde_rec.getActualTCPPose()#[0:3]#robot.fkine(robot.q).t
        goalPose = target_flange_pose

        #rtde_robot.moveL(np.concatenate([goalPose.t, goalPose.rpy()]).tolist(), 0.1, 0.2)
        
        #rtde_robot.servoL(np.concatenate([goalPose, [0,np.pi,np.pi/2]]).tolist(), 0.1, 0.5, 1.0, 0.1, 300)
        #rtde_robot.servoStop()
        
        for t in range(freq*2):
            # Interpolate between the start and goal positions
            interpolated_pose = point_along_line_3d(StartPose, goalPose.t, t, freq*2)

            # Admittance controller step
            f = sensor.get()[3:6]#np.array([0,0,0])#
            quat = goalPose.UnitQuaternion()
            #quat = (SO3.Rx(0.524)*SO3.Ry(-2.403)*SO3.Rz(2.418)).UnitQuaternion()
            quat_array = np.array([quat.s, quat.v[0], quat.v[1], quat.v[2]])  # Convert to [w, x, y, z]
            adm_controller.step(f, mu, SE3(interpolated_pose).t, quat_array)
            u = adm_controller.get_output()
            output_position = u[0:3]
            output_quat = u[3:7]

            target_pose = SE3(pt.transform_from_pq(np.hstack((output_position, output_quat))))
            
            target_pose = np.concatenate([target_pose.t, target_pose.rpy()])
            
            #rtde_robot.servoL(target_pose.tolist())
            #rtde_robot.moveL(target_pose.tolist(), 0.1, 0.3)
            t_start = rtde_robot.initPeriod()
            rtde_robot.servoL(target_pose.tolist(), 0.1, 0.2, dt, 0.1, 300)
            rtde_robot.waitPeriod(t_start)
            #rtde_robot.servoL(target_pose.tolist(), 0.1, 0.1, 0.008, 0.1, 1000)
            #rtde_robot.waitingForMotion()
            #rtde_robot.servoL(target_pose, speed=0.5, acceleration=0.1)
        rtde_robot.servoStop();
        
        print("Sampling downwards...")

        # Move down till force of 8N is reached
        while abs(force) < max_force:
            current_pose = rtde_rec.getActualTCPPose()
            current_tool_pose = xyzrxryrzToSE3(current_pose) * tool_offset
            target_pose = xyzrxryrzToSE3(current_pose) * tool_offset * SE3(0, 0, 0.01)
            target_flange_pose = target_pose * tool_offset.inv()

            StartPose = current_pose[0:3]
            goalPose = target_flange_pose

            for t in range(int(freq*2)):
                # Interpolate between the start and goal positions
                interpolated_pose = point_along_line_3d(StartPose, goalPose.t, t, int(freq*2))

                # Account for pressure robot has to apply
                if abs(force) < max_force:
                    f = sensor.get()[3:6]
                    force = f[0]
                    #f[0] = 0
                    f = np.array([0,0,0])
                else:
                    #f = sensor.get()[3:6]
                    print("LOL you thought you could go over the limit? :D")
                    rtde_robot.servoStop()
                    break

                # Admittance controller step
                quat = goalPose.UnitQuaternion()
                quat_array = np.array([quat.s, quat.v[0], quat.v[1], quat.v[2]])  # Convert to [w, x, y, z]
                adm_controller.step(f, mu, SE3(interpolated_pose).t, quat_array)
                u = adm_controller.get_output()
                output_position = u[0:3]
                output_quat = u[3:7]

                target_pose = SE3(pt.transform_from_pq(np.hstack((output_position, output_quat))))
                
                target_pose = np.concatenate([target_pose.t, target_pose.rpy()])
            
                #rtde_robot.moveL(target_pose.tolist(), 0.1, 0.3)
                t_start = rtde_robot.initPeriod()
                rtde_robot.servoL(target_pose.tolist(), 0.1, 0.05, dt, 0.1, 300)
                rtde_robot.waitPeriod(t_start)
                #rtde_robot.servoL(target_pose.tolist(), 0.1, 0.2, 0.0, 0.1, 300)

                tool_position = (xyzrxryrzToSE3(rtde_rec.getActualTCPPose()) * tool_offset).t
                box_position = box_plane.t
                
                if abs(force) > 0 and contactPos is None:
                    contactPos = tool_position[2]
        rtde_robot.servoStop()
        stopPos = tool_position[2]
        stifness = abs(force) / (contactPos - stopPos)
        force = 0

        # Update the optimizer with the new sample and its corresponding force
        optimizer.update_samples(sample, stifness)

        print(f"Sampled point moving up...")

        # Move back up to beginPose
        current_pose = rtde_rec.getActualTCPPose() #* robot.tool
        target_pose = xyzrxryrzToSE3(current_pose) * tool_offset * SE3(0, 0, -0.02)
        target_flange_pose = target_pose * tool_offset.inv()
        
        StartPose = current_pose[0:3]
        goalPose = target_flange_pose

        for t in range(freq):
            # Interpolate between the start and goal positions
            interpolated_pose = point_along_line_3d(StartPose, goalPose.t, t, freq)

            # Admittance controller step
            f = sensor.get()[3:6]
            quat = goalPose.UnitQuaternion()
            quat_array = np.array([quat.s, quat.v[0], quat.v[1], quat.v[2]])  # Convert to [w, x, y, z]
            adm_controller.step(f, mu, SE3(interpolated_pose).t, quat_array)
            u = adm_controller.get_output()
            output_position = u[0:3]
            output_quat = u[3:7]

            target_pose = SE3(pt.transform_from_pq(np.hstack((output_position, output_quat))))
            target_pose = np.concatenate([target_pose.t, target_pose.rpy()])
            
            #rtde_robot.servoL(target_pose.tolist())
            t_start = rtde_robot.initPeriod()
            rtde_robot.servoL(target_pose.tolist(), 0.1, 0.05, dt, 0.1, 300)
            rtde_robot.waitPeriod(t_start)
        rtde_robot.servoStop()
            
    
    print("Finsihed sampling the first 5 points")

    #----------------------------------------------------------------------------------------
    
    while optimizer.get_number_of_samples() < 12:
        print(f"Current number of samples: {optimizer.get_number_of_samples()}")
        # Get the next sample point
        next_sample = optimizer.get_next_sample()
        # Convert coordinate to robot coordinate
        target_pose = xyToRobotPose(next_sample)
        begin_pose = target_pose
        target_flange_pose = target_pose * tool_offset.inv()

        # Compute the trajectory from the start configuration to the target configuration
        StartPose = rtde_rec.getActualTCPPose()#[0:3]#robot.fkine(robot.q).t
        goalPose = target_flange_pose

        #rtde_robot.moveL(np.concatenate([goalPose.t, goalPose.rpy()]).tolist(), 0.1, 0.2)
        
        #rtde_robot.servoL(np.concatenate([goalPose, [0,np.pi,np.pi/2]]).tolist(), 0.1, 0.5, 1.0, 0.1, 300)
        #rtde_robot.servoStop()
        
        for t in range(freq*2):
            # Interpolate between the start and goal positions
            interpolated_pose = point_along_line_3d(StartPose, goalPose.t, t, freq*2)

            # Admittance controller step
            f = sensor.get()[3:6]#np.array([0,0,0])#
            quat = goalPose.UnitQuaternion()
            #quat = (SO3.Rx(0.524)*SO3.Ry(-2.403)*SO3.Rz(2.418)).UnitQuaternion()
            quat_array = np.array([quat.s, quat.v[0], quat.v[1], quat.v[2]])  # Convert to [w, x, y, z]
            adm_controller.step(f, mu, SE3(interpolated_pose).t, quat_array)
            u = adm_controller.get_output()
            output_position = u[0:3]
            output_quat = u[3:7]

            target_pose = SE3(pt.transform_from_pq(np.hstack((output_position, output_quat))))
            
            target_pose = np.concatenate([target_pose.t, target_pose.rpy()])
            
            #rtde_robot.servoL(target_pose.tolist())
            #rtde_robot.moveL(target_pose.tolist(), 0.1, 0.3)
            t_start = rtde_robot.initPeriod()
            rtde_robot.servoL(target_pose.tolist(), 0.1, 0.2, dt, 0.1, 300)
            rtde_robot.waitPeriod(t_start)
            #rtde_robot.servoL(target_pose.tolist(), 0.1, 0.1, 0.008, 0.1, 1000)
            #rtde_robot.waitingForMotion()
            #rtde_robot.servoL(target_pose, speed=0.5, acceleration=0.1)
        rtde_robot.servoStop();
        
        print("Sampling downwards...")

        # Move down till force of 8N is reached
        while abs(force) < max_force:
            current_pose = rtde_rec.getActualTCPPose()
            current_tool_pose = xyzrxryrzToSE3(current_pose) * tool_offset
            target_pose = xyzrxryrzToSE3(current_pose) * tool_offset * SE3(0, 0, 0.01)
            target_flange_pose = target_pose * tool_offset.inv()

            StartPose = current_pose[0:3]
            goalPose = target_flange_pose

            for t in range(int(freq*2)):
                # Interpolate between the start and goal positions
                interpolated_pose = point_along_line_3d(StartPose, goalPose.t, t, int(freq*2))

                # Account for pressure robot has to apply
                if abs(force) < max_force:
                    f = sensor.get()[3:6]
                    force = f[0]
                    #f[0] = 0
                    f = np.array([0,0,0])
                else:
                    #f = sensor.get()[3:6]
                    print("LOL you thought you could go over the limit? :D")
                    rtde_robot.servoStop()
                    break

                # Admittance controller step
                quat = goalPose.UnitQuaternion()
                quat_array = np.array([quat.s, quat.v[0], quat.v[1], quat.v[2]])  # Convert to [w, x, y, z]
                adm_controller.step(f, mu, SE3(interpolated_pose).t, quat_array)
                u = adm_controller.get_output()
                output_position = u[0:3]
                output_quat = u[3:7]

                target_pose = SE3(pt.transform_from_pq(np.hstack((output_position, output_quat))))
                
                target_pose = np.concatenate([target_pose.t, target_pose.rpy()])
            
                #rtde_robot.moveL(target_pose.tolist(), 0.1, 0.3)
                t_start = rtde_robot.initPeriod()
                rtde_robot.servoL(target_pose.tolist(), 0.1, 0.05, dt, 0.1, 300)
                rtde_robot.waitPeriod(t_start)
                #rtde_robot.servoL(target_pose.tolist(), 0.1, 0.2, 0.0, 0.1, 300)

                tool_position = (xyzrxryrzToSE3(rtde_rec.getActualTCPPose()) * tool_offset).t
                box_position = box_plane.t
                
                if abs(force) > 0 and contactPos is None:
                    contactPos = tool_position[2]
        rtde_robot.servoStop()
        stopPos = tool_position[2]
        stifness = abs(force) / (contactPos - stopPos)
        force = 0

        # Update the optimizer with the new sample and its corresponding force
        optimizer.update_samples(sample, stifness)

        print(f"Sampled point moving up...")

        # Move back up to beginPose
        current_pose = rtde_rec.getActualTCPPose() #* robot.tool
        target_pose = xyzrxryrzToSE3(current_pose) * tool_offset * SE3(0, 0, -0.03)
        target_flange_pose = target_pose * tool_offset.inv()
        
        StartPose = current_pose[0:3]
        goalPose = target_flange_pose

        for t in range(freq):
            # Interpolate between the start and goal positions
            interpolated_pose = point_along_line_3d(StartPose, goalPose.t, t, freq)

            # Admittance controller step
            f = sensor.get()[3:6]
            quat = goalPose.UnitQuaternion()
            quat_array = np.array([quat.s, quat.v[0], quat.v[1], quat.v[2]])  # Convert to [w, x, y, z]
            adm_controller.step(f, mu, SE3(interpolated_pose).t, quat_array)
            u = adm_controller.get_output()
            output_position = u[0:3]
            output_quat = u[3:7]

            target_pose = SE3(pt.transform_from_pq(np.hstack((output_position, output_quat))))
            target_pose = np.concatenate([target_pose.t, target_pose.rpy()])
            
            #rtde_robot.servoL(target_pose.tolist())
            t_start = rtde_robot.initPeriod()
            rtde_robot.servoL(target_pose.tolist(), 0.1, 0.05, dt, 0.1, 300)
            rtde_robot.waitPeriod(t_start)
        rtde_robot.servoStop()

    print(f"Sampled all {optimizer.get_number_of_samples() + 1} points. Plotting the results...")
    optimizer.plot_optimization()



    
                
                    

            



            

        
        