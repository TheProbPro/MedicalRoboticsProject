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

# Admittance controller imports
from sdu_controllers import AdmittanceControllerPosition

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

def force_calculator(tool_position, box_position, box_size):
    F = 0
    # Tissue stiffness constant
    k_tissue = 1000
    # Calculate the box boundaries
    box_min = box_position - box_size / 2
    box_max = box_position + box_size / 2

    # Check if the tool is inside the box
    if np.all(tool_position >= box_min) and np.all(tool_position <= box_max):
        # Calculate penetration depth
        penetration_depth = box_max[2] - tool_position[2]
        # Calculate force based on penetration depth
        if penetration_depth > 0:
            F = k_tissue * penetration_depth
            #print(f"Tool is inside the box. Penetration depth: {penetration_depth:.4f} m, Force: {F:.2f} N")
    # Return the force value
    return F

def xyToRobotPose(xy, robot):
    x, y = xy
    z = 0.1
    robot_pose = SE3(x, y, z) * SE3(SO3.Rx(np.pi)) * robot.base
    return robot_pose

def xyzToRobotPose(xyz, robot):
    x, y, z = xyz
    robot_pose = SE3(x, y, z) * SE3(SO3.Rx(np.pi)) * robot.base
    return robot_pose

def find_valid_trajectory(robot, target_pose, q_start):
    # Get the z-coordinate of the robot base
    ground_z = robot.base.t[2]

    # Loop until a valid trajectory is found
    while True:
        # Account for tool offset: compute pose for the flange
        target_flange_pose = target_pose * robot.tool.inv()
        # Compute the inverse kinematics
        q, success, iter, searches, residual = robot.ik_LM(target_flange_pose, q0=robot.q)
        # Check if the IK solution is valid
        if not success:
            raise RuntimeError("IK failed to find a solution.")

        # Compute the trajectory from the start configuration to the target configuration
        q_target = q
        trajectory = rtb.jtraj(q_start, q_target, 500)

        is_valid = True

        # Check if the trajectory is valid
        for q in trajectory.q:
            # Compute the forward kinematics for all joints
            fk = robot.fkine_all(q)
            # Check if any joint is below the ground level
            if any(joint_pose.t[2] < ground_z for joint_pose in fk):
                is_valid = False
                break
            # if fk[2].t[2] < fk[1].t[2]:
            #     is_valid = False
            #     break
            # Make sure the elbow joint (joint 3) is above the shoulder joint (joint 2)
            if fk[3].t[2] < fk[2].t[2]:
                is_valid = False
                break
            # Compute the Jacobian matrix for the current configuration and check for singularities
            J = robot.jacob0(q)
            if np.linalg.matrix_rank(J) < 6:
                is_valid = False
                break
        # If the trajectory is valid, return it
        if is_valid:
            return trajectory

def execute_trajectory(robot, trajectory, admittanceController, box_plane, box_size, dt):
    force = 0
    f = np.zeros(3)
    mu = np.zeros(3)
    quat_init = SE3(robot.fkine(robot.q)).UnitQuaternion()
    quat_init = np.array([quat_init.s, quat_init.v[0], quat_init.v[1], quat_init.v[2]])  # [w, x, y, z]
    # execute the trajectory to the point
    for q in trajectory.q:
        # Get the target pose
        target_pose = robot.fkine(q) * robot.tool
        #print(f"target pose: {target_pose}")
        
        # step the admittance controller
        f[2] = force
        admittanceController.step(f, mu, target_pose.t, quat_init)
        u = admittanceController.get_output()
        output_position = u[0:3]
        output_quat = u[3:7]

        # rotate output from tip to TCP before sending it to the robot
        target_pose = SE3(pt.transform_from_pq(np.hstack((output_position, output_quat))))
        #print(f"target pose: {target_pose}")
        # Account for tool offset: compute pose for the flange
        target_flange_pose = target_pose * robot.tool.inv()
        # Compute the inverse kinematics
        q, success, iter, searches, residual = robot.ik_LM(target_flange_pose, q0=robot.q, end='tool0')
        # Check if the IK solution is valid
        if not success:
            raise RuntimeError("IK failed to find a solution.")

        # Compute the trajectory from the start configuration to the target configuration
        robot.q = q
        env.step(dt)
        
    contactPos = None
    stopPos = None

    # Move down till force of 8N is reached
    while True:
        # Get the current tool pose
        current_tool_pose = robot.fkine(robot.q) * robot.tool
        # Get the target pose
        target_pose = SE3(current_tool_pose.t) * SE3(SO3.Rx(np.pi)) * SE3(0, 0, 0.01)
        # Find a valid trajectory to the target pose   
        traj = find_valid_trajectory(robot, target_pose, robot.q)

        # Execute the trajectory
        for q in traj.q:
            # Get the target pose
            target_pose = robot.fkine(q) * robot.tool
            #quat_init = target_pose.UnitQuaternion()
            #quat_init = np.array([quat_init.s, quat_init.v[0], quat_init.v[1], quat_init.v[2]])  # [w, x, y, z]
            # Step the admittance controller
            print(f"target pose: {target_pose}")
            if force < 8.0:
                f[2] = 0
            else:
                f[2] = force

            admittanceController.step(f, mu, target_pose.t, SO3.Rx(np.pi).UnitQuaternion()) # TODO: I do not know if i did this correct
            u = admittanceController.get_output()
            output_position = u[0:3]
            output_quat = u[3:7]

            # rotate output from tip to TCP before sending it to the robot
            target_pose = SE3(pt.transform_from_pq(np.hstack((output_position, output_quat))))
            #print(f"target pose: {target_pose}")
            # Account for tool offset: compute pose for the flange
            target_flange_pose = target_pose * robot.tool.inv()
            # Compute the inverse kinematics
            ikq, success, iter, searches, residual = robot.ik_LM(target_flange_pose, q0=robot.q, end='tool0')
            # Check if the IK solution is valid
            # if not success:
            #     raise RuntimeError("IK failed to find a solution.")

            # Compute the trajectory from the start configuration to the target configuration
            robot.q = ikq
            env.step(dt)

            tool_position = (robot.fkine(q) * robot.tool).t
            box_position = box_plane.t
            force = force_calculator(tool_position, box_position, box_size)

            if force > 0 and contactPos is None:
                contactPos = tool_position[2]

            if force >= 8.0:
                stopPos = tool_position[2]
                break
            
        if stopPos is not None:
            break

    # Move back up
    current_tool_pose = robot.fkine(robot.q) * robot.tool
    target_pose = SE3(current_tool_pose.t) * SE3(SO3.Rx(np.pi)) * SE3(0, 0, -0.02)

    traj = find_valid_trajectory(robot, target_pose, robot.q)
    for q in traj.q:
        # Get the target pose
        target_pose = robot.fkine(q) * robot.tool
        # Step the admittance controller
        f[2] = force
        admittanceController.step(f, mu, target_pose.t, quat_init)
        u = admittanceController.get_output()
        output_position = u[0:3]
        output_quat = u[3:7]
        # Rotate output from tip to TCP before sending it to the robot
        target_pose = SE3(pt.transform_from_pq(np.hstack((output_position, output_quat))))
        #print(f"target pose: {target_pose}")
        # Account for tool offset: compute pose for the flange
        target_flange_pose = target_pose * robot.tool.inv()
        # Compute the inverse kinematics
        q, success, iter, searches, residual = robot.ik_LM(target_flange_pose, q0=robot.q, end='tool0')
        # Check if the IK solution is valid
        if not success:
            raise RuntimeError("IK failed to find a solution.")

        # Compute the trajectory from the start configuration to the target configuration
        robot.q = q
            
        env.step(dt)

    return (contactPos - stopPos) * force

if __name__ == '__main__':
    # Initialize robot parameters
    freq = 500
    dt = 1 / freq

    # Initialize the admittance controller
    adm_controller = AdmittanceControllerPosition(freq)
    adm_controller.set_mass_matrix_position(np.identity(3) * 22.5)
    adm_controller.set_stiffness_matrix_position(np.identity(3) * 54)
    adm_controller.set_damping_matrix_position(np.identity(3) * 160)

    adm_controller.set_mass_matrix_orientation(np.identity(3) * 0.25)
    adm_controller.set_stiffness_matrix_orientation(np.identity(3) * 100)
    adm_controller.set_damping_matrix_orientation(np.identity(3) * 100)

    # Initialize the robot and set its initial configuration
    robot = rtb.models.UR3()
    robot.tool = SE3(0.03, 0, 0.075)  # Tool offset
    q_start = np.array([0, -np.pi / 4, np.pi / 4, -np.pi / 4, np.pi / 2, 0])
    #q_start = np.array([np.pi, -np.pi / 4, np.pi / 4, -np.pi / 4, np.pi / 2, 0])
    robot.q = q_start
    
    #Define frames
    T_base_tcp = SE3(robot.fkine(robot.q))
    T_tcp_tip = robot.tool
    T_tip_tcp = T_tcp_tip.inv()
    T_base_tip = T_base_tcp * T_tip_tcp
    T_base_tip_init = SE3(T_base_tip)

    # Initialize the environment
    env = Swift()

    # Create the floor and box planes
    floor_plane = sg.Cuboid([10, 10, 0.01], pose=robot.base, collision=True)
    box_plane = SE3(0.4, 0, 0)
    box_size = np.array([0.1, 0.1, 0.1])
    box = sg.Cuboid(box_size, pose=box_plane, collision=True)

    # add the robot, box, and floor to the environment
    env.launch(frequency=freq, realtime=True)
    env.add(robot, collision_alpha=1.0)
    env.add(box, collision_alpha=1.0)
    env.add(floor_plane, collision_alpha=0.1)
    env.step()

    time.sleep(1)

    # Initialize the optimizer
    optimizer = BayesianOptimizer3D(box_size, box_plane.t)
    # Generate the first set of random samples
    samples = optimizer.init_random_samples()

    # Get the z-coordinate of the robot base
    ground_z = robot.base.t[2]

    f = np.zeros(3)
    mu = np.zeros(3)
    quat_init = SE3(robot.fkine(robot.q)).UnitQuaternion()
    quat_init = np.array([quat_init.s, quat_init.v[0], quat_init.v[1], quat_init.v[2]])
    
    force = 0
    contactPos = None

    # Run for x iterations
    for sample in samples:
        # Convert 2D to Robot coordinate
        target_pose = xyToRobotPose(sample, robot)
        begin_pose = target_pose
        target_flange_pose = target_pose * robot.tool.inv()

        # Compute the trajectory from the start configuration to the target configuration
        StartPose = robot.fkine(robot.q).t
        goalPose = target_flange_pose.t

        for t in range(freq):
            # Interpolate between the start and goal positions
            interpolated_pose = point_along_line_3d(StartPose, goalPose, t, freq)

            # Admittance controller step
            adm_controller.step(f, mu, SE3(interpolated_pose).t, quat_init)
            u = adm_controller.get_output()
            output_position = SE3(u[0:3])
            
            valid_trajectory = False
            q, success, iter, searches, residual = robot.ik_LM(output_position, q0=robot.q)
            # Check if the IK solution is valid
            if not success:
                raise RuntimeError("IK failed to find a solution.")
            
            # TODO: Implement the validity checks
            # while not valid_trajectory:
            #     # Inverse kinematics
            #     q, success, iter, searches, residual = robot.ik_LM(output_position, q0=robot.q)
            #     # Check if the IK solution is valid
            #     if not success:
            #         raise RuntimeError("IK failed to find a solution.")
                
            #     # Check if any joint is below the ground level
            #     fk = robot.fkine_all(q)
            #     #print(f"fk[0]: {fk[0].t}, fk[1]: {fk[1].t}, fk[2]: {fk[2].t}, fk[3]: {fk[3].t}, fk[4]: {fk[4].t}, fk[5]: {fk[5].t}, fk[6]: {fk[6].t}")
            #     #print(f"fk: {fk.t}")
            #     if not all(joint_pose.t[2] >= ground_z for joint_pose in fk):
            #         valid_trajectory = False
            #     elif fk[3].t[2] <= fk[2].t[2]:
            #         valid_trajectory = False
            #     else:
            #         valid_trajectory = True
                
            robot.q = q
            env.step(dt)
        
        # Move down till force of 8N is reached
        # TODO: Fix the instant flip
        while force < 8.0:
            current_pose = robot.fkine(robot.q)
            current_tool_pose = current_pose * robot.tool
            target_pose = SE3(current_pose.t) * SE3(SO3.Rx(np.pi)) * SE3(0, 0, 0.01) * robot.tool
            
            quat = target_pose.R
            target_flange_pose = target_pose * robot.tool.inv()

            StartPose = robot.fkine(robot.q).t
            goalPose = target_flange_pose.t
            #goalPose = target_pose.t

            for t in range(int(freq/2)):
                # Interpolate between the start and goal positions
                interpolated_pose = point_along_line_3d(StartPose, goalPose, t, int(freq/2))

                # Account for pressure robot has to apply
                if force < 8.0:
                    f[2] = 0
                else:
                    f[2] = force

                # Admittance controller step
                quat = SO3.Rx(np.pi).UnitQuaternion()
                quat_array = np.array([quat.s, quat.v[0], quat.v[1], quat.v[2]])  # Convert to [w, x, y, z]
                adm_controller.step(f, mu, SE3(interpolated_pose).t, quat_array)
                u = adm_controller.get_output()
                output_position = u[0:3]
                output_quat = u[3:7]

                target_pose = SE3(pt.transform_from_pq(np.hstack((output_position, output_quat))))
                
            
                valid_trajectory = False
                q, success, iter, searches, residual = robot.ik_LM(target_pose, q0=robot.q)
                # Check if the IK solution is valid
                if not success:
                    raise RuntimeError("IK failed to find a solution.")

                #TODO: Implement the validity checks

                robot.q = q
                env.step(dt)

                tool_position = (robot.fkine(q) * robot.tool).t
                box_position = box_plane.t
                force = force_calculator(tool_position, box_position, box_size)
                #print(f"force: {force}")

                if force > 0 and contactPos is None:
                    contactPos = tool_position[2]

        stopPos = tool_position[2]
        stifness = (contactPos - stopPos) * force
        force = 0

        # Update the optimizer with the new sample and its corresponding force
        optimizer.update_samples(sample, stifness)

        # Move back up to beginPose
        current_tool_pose = robot.fkine(robot.q) #* robot.tool
        target_pose = SE3(current_tool_pose.t) * SE3(SO3.Rx(np.pi)) * SE3(0, 0, -0.02)
        StartPose = robot.fkine(robot.q).t
        goalPose = target_pose.t

        for t in range(freq):
            # Interpolate between the start and goal positions
            interpolated_pose = point_along_line_3d(StartPose, goalPose, t, freq)

            # Admittance controller step
            f[2] = force
            quat = SO3.Rx(np.pi).UnitQuaternion()
            quat_array = np.array([quat.s, quat.v[0], quat.v[1], quat.v[2]])  # Convert to [w, x, y, z]
            adm_controller.step(f, mu, SE3(interpolated_pose).t, quat_array)
            u = adm_controller.get_output()
            output_position = u[0:3]
            output_quat = u[3:7]

            target_pose = SE3(pt.transform_from_pq(np.hstack((output_position, output_quat))))
            
            valid_trajectory = False
            q, success, iter, searches, residual = robot.ik_LM(target_pose, q0=robot.q)
            # Check if the IK solution is valid
            if not success:
                raise RuntimeError("IK failed to find a solution.")
            
            #TODO: Implement the validity checks

            robot.q = q
            env.step(dt)






    
                
                    

            



            

        
        