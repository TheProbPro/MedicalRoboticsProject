import roboticstoolbox as rtb
import numpy as np
from spatialmath import SE3, SO3
from spatialmath.base import *
import logging
import time
from roboticstoolbox.backends.swift import Swift
import spatialgeometry as sg

# Admittance controller imports
from sdu_controllers import AdmittanceControllerPosition

# Import BO libraries
from BaysianOptimization.Optimizer import BayesianOptimizer3D

_logger = logging.getLogger('Palpation_robot')

# TODO: Add robot.tool to all poses, implement Admittance controller
# the following might be fixed: (Fix elbow joint pointing downwards, robot doing large rotations (180*), toolflange rotating)

def force_calculator(tool_position, box_position, box_size):
    """
    This function calculates the force applied to the endeffector of the robot, depending on how much it has penetrated the box phantom.
    """
    F = 0
    # Define the tissue constants
    k_tissue = 1000  # Tissue stiffness estimate (i came up with this number)

    # Check if the tool is inside the box
    box_min = box_position - box_size / 2  # Minimum corner of the box
    box_max = box_position + box_size / 2  # Maximum corner of the box

    if np.all(tool_position >= box_min) and np.all(tool_position <= box_max):
        # Tool is inside the box
        penetration_depth = box_max[2] - tool_position[2]  # Penetration along the z-axis
        if penetration_depth > 0:
            # Calculate the force using Hooke's law
            F = k_tissue * penetration_depth
            print(f"Tool is inside the box. Penetration depth: {penetration_depth:.4f} m, Force: {F:.2f} N")
    
    return F

def xyToRobotPose(xy, robot):
    """
    This function converts a 2D point in the box plane to a 3D pose for the robot.
    """
    x, y = xy
    z = 0.1
    # Convert to robot pose
    robot_pose = SE3(x, y, z) * robot.base
    return robot_pose

def find_valid_trajectory(robot, target_pose, q_start):
    """
    This function finds a valid trajectory for the robot to reach the target pose without going below z=0.
    """

    ground_z = robot.base.t[2]

    while True:
        #ik_solution = robot.ikine_LM(target_pose)

        q, success, iter, searches, residual = robot.ik_LM(target_pose, q0=robot.q)

        if not success:
            raise RuntimeError("IK failed to find a solution.")
    
        q_target = q
        trajectory = rtb.jtraj(q_start, q_target, 1000)

        is_valid = True

        for q in trajectory.q:
            fk = robot.fkine_all(q)
            # check if any joint is below the ground
            if any(joint_pose.t[2] < ground_z for joint_pose in fk):
                is_valid = False
                break  # A joint is below the ground

            # Check if the elbow joint is above the shoulder joint
            if fk[2].t[2] < fk[1].t[2]:
                is_valid = False
                break

            # Check for singularities
            J = robot.jacob0(q)
            if np.linalg.matrix_rank(J) < 6:  # Or use condition number check
                is_valid = False
                break  # Singularity or near singularity

        if is_valid:
            return trajectory

    
def execute_trajectory(robot, trajectory, box_plane, box_size, dt):
    """
    This function executes the trajectory and calculates the force applied to the end-effector.
    """
    # Robot executes trajectory
    for q in trajectory.q:
        robot.q = q
        # Simulate environment step
        env.step(dt)

    # Initialize force variables
    force = 0
    contactPos = None
    stopPos = None

    # Make robot move down the z-axis till force is above 8N
    while True:
        #target_pose = robot.fkine(robot.q) + SE3(0, 0, -0.01)  # Move down by 1 cm
        target_pose = SE3(robot.fkine(robot.q).t) * SE3(SO3.Rx(np.pi)) * SE3(0, 0, 0.01)  # Move down by 1 cm
        # TODO: Alternatively use cartesian pathplanning
        traj = find_valid_trajectory(robot, target_pose, robot.q)
        for q in traj.q:
            robot.q = q

            # Simulate environment step
            env.step(dt)

            # Get the current end-effector position
            tool_position = robot.fkine(q).t

            # Calculate the force applied to the end-effector
            box_position = box_plane.t
            force = force_calculator(tool_position, box_position, box_size)

            if force > 0 and contactPos is None:
                contactPos = tool_position[2]

            if force >= 8.0:
                stopPos = tool_position[2]
                break
        if stopPos is not None:
            break
    
    # Move up again
    target_pose = SE3(robot.fkine(robot.q).t) * SE3(SO3.Rx(np.pi)) * SE3(0, 0, -0.02)  # Move up by 2 cm
    traj = find_valid_trajectory(robot, target_pose, robot.q)
    for q in traj.q:
        robot.q = q

        # Simulate environment step
        env.step(dt)

    return (contactPos- stopPos) * force * 100


if __name__ == '__main__':
    # Setup robot parameters
    freq = 500
    dt = 1/freq

    # Initialize robot
    robot = rtb.models.UR3()
    robot.tool = SE3(0.03, 0, 0.075) # adds 30mm to the axis of the last joint and 75mm to the z-axis
    q_start = np.array([0, -np.pi / 4, np.pi / 4, -np.pi / 4, np.pi / 2, 0])
    robot.q = q_start

    # Setup environment
    env = Swift()

    # define floor/table plane
    floor_plane = sg.Cuboid([10,10,0.01], pose=robot.base, collision=True)

    # define the place the cube should be on
    box_plane = SE3(0.4,0,0)

    # Create box
    box_size = np.array([0.1, 0.1, 0.1])
    box = sg.Cuboid(box_size, pose=box_plane, collision=True)

    # Add robot and box to environment
    env.launch(frequency=freq, realtime=True)
    env.add(robot, collision_alpha=1.0)
    env.add(box, collision_alpha=1.0)
    env.add(floor_plane, collision_alpha=0.1)
    env.step()

    time.sleep(1)

    # Initialize the Bayesian optimizer
    optimizer = BayesianOptimizer3D(box_size, box_plane.t)
    # Get initial samples
    samples = optimizer.init_random_samples()

    # Simulate sampling
    while optimizer.get_number_of_samples() < 50:#50:
        i = optimizer.get_number_of_samples()
        if i < 5:
            print(f"Sample {i} of 5: {samples[i]}")
            traj = find_valid_trajectory(robot, xyToRobotPose(samples[i], robot), robot.q)
            stiffness = execute_trajectory(robot, traj, box_plane, box_size, dt)
            print(f"Updating sample {i} with sample {samples[i]}, stiffness {stiffness}")
            optimizer.update_samples(samples[i], stiffness)
        else:
            # Get the next sample point
            next_sample = optimizer.get_next_sample()
            print(f"Sample {i} of 50: {next_sample}")
            # Execute the trajectory for the next sample
            traj = find_valid_trajectory(robot, xyToRobotPose(next_sample, robot), robot.q)
            stiffness = execute_trajectory(robot, traj, box_plane, box_size, dt)
            print(f"Updating sample {i} with sample {next_sample}, stiffness {stiffness}")
            optimizer.update_samples(next_sample, stiffness)

    print("finish sampling")

    # Plot the optimization results
    optimizer.plot_optimization()


# import roboticstoolbox as rtb
# import numpy as np
# from spatialmath import SE3, SO3
# from spatialmath.base import *
# import logging
# import time
# from roboticstoolbox.backends.swift import Swift
# import spatialgeometry as sg

# # Admittance controller imports
# from sdu_controllers import AdmittanceControllerPosition

# # Import BO libraries
# from BaysianOptimization.Optimizer import BayesianOptimizer3D

# _logger = logging.getLogger('Palpation_robot')

# def force_calculator(tool_position, box_position, box_size):
#     """
#     This function calculates the force applied to the endeffector of the robot, depending on how much it has penetrated the box phantom.
#     """
#     F = 0
#     # Define the tissue constants
#     k_tissue = 1000  # Tissue stiffness estimate (i came up with this number)

#     # Check if the tool is inside the box
#     box_min = box_position - box_size / 2  # Minimum corner of the box
#     box_max = box_position + box_size / 2  # Maximum corner of the box

#     if np.all(tool_position >= box_min) and np.all(tool_position <= box_max):
#         # Tool is inside the box
#         penetration_depth = box_max[2] - tool_position[2]  # Penetration along the z-axis
#         if penetration_depth > 0:
#             # Calculate the force using Hooke's law
#             F = k_tissue * penetration_depth
#             print(f"Tool is inside the box. Penetration depth: {penetration_depth:.4f} m, Force: {F:.2f} N")
    
#     return F

# def find_valid_trajectory(robot, target_pose, q_start):
#     """
#     This function finds a valid trajectory for the robot to reach the target pose without going below z=0.
#     """
#     while True:
#         #ik_solution = robot.ikine_LM(target_pose)
#         q, success, iter, searches, residual = robot.ik_LM(target_pose)

#         if not success:
#             raise RuntimeError("IK failed to find a solution.")
    
#         q_target = q
#         trajectory = rtb.jtraj(q_start, q_target, 1000)

#         is_valid = True

#         for q in trajectory.q:
#             fk = robot.fkine_all(q)
#             if any(joint_pose.t[2] < 0 for joint_pose in fk):
#                 is_valid = False
#                 break  # A joint is below the ground

#             J = robot.jacob0(q)
#             if np.linalg.matrix_rank(J) < 6:  # Or use condition number check
#                 is_valid = False
#                 break  # Singularity or near singularity

#         if is_valid:
#             return trajectory


# # def find_valid_trajectory(robot, target_pose, q_start):
# #     """
# #     This function finds a valid trajectory for the robot to reach the target pose without going below z=0.
# #     """
# #     valid_trajectory = False
# #     trajectory = None
# #     while not valid_trajectory:
# #         # Plan trajectory
# #         ik_solution = robot.ikine_LM(target_pose)

# #         if not ik_solution.success:
# #             raise RuntimeError("IK failed to find a solution.")
    
# #         q_target = ik_solution.q
# #         trajectory = rtb.jtraj(q_start, q_target, 1000)

# #         # Check if any joint goes below z=0
# #         for q in trajectory.q:
# #             fk = robot.fkine_all(q)  # Compute forward kinematics
# #             for joint_pose in fk:
# #                 if joint_pose.t[2] < 0:  # Check if z-coordinate is below 0
# #                     #print("Joint below z=0 detected!")
# #                     break
            
# #             # Check for singularity
# #             J = robot.jacob0(q)  # Compute the Jacobian at the current configuration
# #             if np.linalg.det(J) == 0:  # High condition number indicates a singularity
# #                 #print("Singularity detected!")
# #                 break
            
# #             valid_trajectory = True

# #     return trajectory
    
# def execute_trajectory(robot, trajectory, box_plane, box_size, dt):
#     """
#     This function executes the trajectory and calculates the force applied to the end-effector.
#     """
#     # Robot executes trajectory
#     for q in trajectory.q:
#         robot.q = q
#         # Simulate environment step
#         env.step(dt)

#     # Initialize force variables
#     force = 0
#     contactPos = None
#     stopPos = None

#     # Make robot move down the z-axis till force is above 8N
#     while True:
#         target_pose = robot.fkine(robot.q) + SE3(0, 0, -0.01)  # Move down by 1 cm
#         traj = find_valid_trajectory(robot, target_pose, robot.q)
#         for q in traj.q:
#             robot.q = q

#             # Simulate environment step
#             env.step(dt)

#             # Get the current end-effector position
#             tool_position = robot.fkine(q).t

#             # Calculate the force applied to the end-effector
#             box_position = box_plane.t
#             force = force_calculator(tool_position, box_position, box_size)

#             if force > 0 and contactPos is None:
#                 contactPos = tool_position[2]

#             if force >= 8.0:
#                 stopPos = tool_position[2]
#                 return (stopPos - contactPos) * force

#         # Get the current end-effector position


#     # for q in trajectory.q:
#     #     robot.q = q

#     #     # Get the current end-effector position
#     #     tool_position = robot.fkine(q).t

#     #     # Calculate the force applied to the end-effector
#     #     box_position = box_plane.t
#     #     force = force_calculator(tool_position, box_position, box_size)

#     #     if force > 0 and contactPos is None:
#     #         contactPos = tool_position

#     #     if force > 8.0:
#     #         stopPos = tool_position
#     #         break

#     #     # Simulate environment step
#     #     env.step(dt)
#     # if contactPos is None and stopPos is None:
#     #     return 0
#     # else:
#     #     return (stopPos - contactPos) * force

# def xyToRobotPose(xy, robot):
#     """
#     This function converts a 2D point in the box plane to a 3D pose for the robot.
#     """
#     # TODO: This is not done
#     x = xy[0]
#     y = xy[1]
#     z = 0.1
#     # Convert to robot pose
#     tcp_orientation = SO3.Rx(np.pi)  # Point TCP Z down
#     robot_pose = SE3(x, y, z) * robot.base# * SE3(tcp_orientation)
#     return robot_pose


# if __name__ == '__main__':
#     # Setup robot parameters
#     freq = 500
#     dt = 1/freq

#     # Initialize robot
#     robot = rtb.models.UR3()
#     robot.tool = SE3(0, 0, 0.1)
#     q_start = np.array([0, -np.pi / 4, np.pi / 4, -np.pi / 4, np.pi / 2, 0])
#     robot.q = q_start

#     # Setup environment
#     env = Swift()

#     # define floor/table plane
#     floor_plane = sg.Cuboid([10,10,0.01], pose=robot.base, collision=True)

#     # define the place the cube should be on
#     box_plane = SE3(0.4,0,0)

#     # Create box
#     box_size = np.array([0.1, 0.1, 0.1])
#     box = sg.Cuboid(box_size, pose=box_plane, collision=True)

#     # Add robot and box to environment
#     env.launch(frequency=freq, realtime=True)
#     env.add(robot, collision_alpha=1.0)
#     env.add(box, collision_alpha=1.0)
#     env.add(floor_plane, collision_alpha=0.1)
#     env.step()

#     time.sleep(1)

#     #define target pose 0.2 above the box and tool oriented to point at the box
#     #orientation = SE3.Rt(SO3.Ry(np.pi/2), [0, 0, 0.1])
#     orientation = SE3.Rt(SO3.Rz(np.pi/2), [0, 0, 0.02])
#     target_pose = box_plane * orientation#SE3.Trans(0, 0, 0.2) * SO3.rpy(0, np.pi/2, np.pi/2)#SE3.Rx(-np.pi / 2) * SE3.Ry(np.pi / 2) * SE3.Rz(np.pi / 2)

#     # Initialize the Bayesian optimizer
#     optimizer = BayesianOptimizer3D(box_size, box_plane.t)
#     # Get initial samples
#     samples = optimizer.init_random_samples()

#     # Simulate sampling
#     while optimizer.get_number_of_samples() < 10:#50:
#         i = optimizer.get_number_of_samples()
#         if i < 5:
#             traj = find_valid_trajectory(robot, xyToRobotPose(samples[i], robot), robot.q)
#             stiffness = execute_trajectory(robot, traj, box_plane, box_size, dt)
#             print(f"Updating samples with sample {samples[i]}, stiffness {stiffness}")
#             optimizer.update_samples(samples[i], stiffness)
#         else:
#             # Get the next sample point
#             next_sample = optimizer.get_next_sample()
#             # Execute the trajectory for the next sample
#             traj = find_valid_trajectory(robot, xyToRobotPose(next_sample, robot), robot.q)
#             stiffness = execute_trajectory(robot, traj, box_plane, box_size, dt)
#             print(f"Updating samples with sample {next_sample}, stiffness {stiffness}")
#             optimizer.update_samples(next_sample, stiffness)

#     print("finish sampling")

#     # Plot the optimization results
#     optimizer.plot_optimization()

    

#     # Check if any joints go below the floor_plane
#     # while True:
#     #     # Plan trajectory
#     #     ik_solution = robot.ikine_LM(target_pose)

#     #     if not ik_solution.success:
#     #         raise RuntimeError("IK failed to find a solution.")
    
#     #     q_target = ik_solution.q
#     #     trajectory = rtb.jtraj(q_start, q_target, 1000)

#     #     # Check if any joint goes below z=0
#     #     valid_trajectory = True
#     #     for q in trajectory.q:
#     #         fk = robot.fkine_all(q)  # Compute forward kinematics
#     #         for joint_pose in fk:
#     #             if joint_pose.t[2] < 0:  # Check if z-coordinate is below 0
#     #                 valid_trajectory = False
#     #                 print("Joint below z=0 detected!")
#     #                 break
            
#     #         # Check for singularity
#     #         J = robot.jacob0(q)  # Compute the Jacobian at the current configuration
#     #         if np.linalg.det(J) == 0:  # High condition number indicates a singularity
#     #             valid_trajectory = False
#     #             print("Singularity detected!")
#     #             break

#     #     if valid_trajectory:
#     #         break  # Exit loop if trajectory is valid
#     #     #else:
#     #     #   # Adjust target pose slightly upward and retry
#     #     #    target_pose = target_pose * SE3.Trans(0, 0, 0.001)

#     # # Execute the valid trajectory
#     # for q in trajectory.q:
#     #     robot.q = q

#     #     # Get the current end-effector position
#     #     tool_position = robot.fkine(q).t

#     #     # Calculate the force applied to the end-effector
#     #     box_position = box_plane.t
#     #     force = force_calculator(tool_position, box_position, box_size)

#     #     #env.step(0.02)
#     #     env.step(dt)

#     # print('Moved to box')
