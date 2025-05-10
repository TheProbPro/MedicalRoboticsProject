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
from BaysianOptimization.Target import target_function2

_logger = logging.getLogger('Palpation_robot')

def force_calculator(tool_position, box_position, box_size):
    F = 0
    k_tissue = 1000
    box_min = box_position - box_size / 2
    box_max = box_position + box_size / 2

    if np.all(tool_position >= box_min) and np.all(tool_position <= box_max):
        penetration_depth = box_max[2] - tool_position[2]
        if penetration_depth > 0:
            F = k_tissue * penetration_depth
            print(f"Tool is inside the box. Penetration depth: {penetration_depth:.4f} m, Force: {F:.2f} N")

    return target_function2(box_position, F)

def xyToRobotPose(xy, robot):
    x, y = xy
    z = 0.1
    robot_pose = SE3(x, y, z) * SE3(SO3.Rx(np.pi)) * robot.base
    return robot_pose

def find_valid_trajectory(robot, target_pose, q_start):
    ground_z = robot.base.t[2]

    while True:
        # Account for tool offset: compute pose for the flange
        target_flange_pose = target_pose * robot.tool.inv()

        q, success, iter, searches, residual = robot.ik_LM(target_flange_pose, q0=robot.q)

        if not success:
            raise RuntimeError("IK failed to find a solution.")

        q_target = q
        trajectory = rtb.jtraj(q_start, q_target, 500)

        is_valid = True

        for q in trajectory.q:
            fk = robot.fkine_all(q)
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
            
            J = robot.jacob0(q)
            if np.linalg.matrix_rank(J) < 6:
                is_valid = False
                break

        if is_valid:
            return trajectory

def execute_trajectory(robot, trajectory, box_plane, box_size, dt):
    for q in trajectory.q:
        robot.q = q
        env.step(dt)

    force = 0
    contactPos = None
    stopPos = None

    while True:
        current_tool_pose = robot.fkine(robot.q) * robot.tool
        target_pose = SE3(current_tool_pose.t) * SE3(SO3.Rx(np.pi)) * SE3(0, 0, 0.01)

        traj = find_valid_trajectory(robot, target_pose, robot.q)

        for q in traj.q:
            robot.q = q
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
        robot.q = q
        env.step(dt)

    return (contactPos - stopPos) * force

if __name__ == '__main__':
    freq = 500
    dt = 1 / freq

    robot = rtb.models.UR3()
    robot.tool = SE3(0.03, 0, 0.075)  # Tool offset
    q_start = np.array([0, -np.pi / 4, np.pi / 4, -np.pi / 4, np.pi / 2, 0])
    robot.q = q_start

    env = Swift()

    floor_plane = sg.Cuboid([10, 10, 0.01], pose=robot.base, collision=True)
    box_plane = SE3(0.4, 0, 0)
    box_size = np.array([0.1, 0.1, 0.1])
    box = sg.Cuboid(box_size, pose=box_plane, collision=True)

    env.launch(frequency=freq, realtime=True)
    env.add(robot, collision_alpha=1.0)
    env.add(box, collision_alpha=1.0)
    env.add(floor_plane, collision_alpha=0.1)
    env.step()

    time.sleep(1)

    optimizer = BayesianOptimizer3D(box_size, box_plane.t)
    samples = optimizer.init_random_samples()

    while optimizer.get_number_of_samples() < 20:
        i = optimizer.get_number_of_samples()
        if i < 5:
            traj = find_valid_trajectory(robot, xyToRobotPose(samples[i], robot), robot.q)
            stiffness = execute_trajectory(robot, traj, box_plane, box_size, dt)
            print(f"Updating sample {i} with sample {samples[i]}, stiffness {stiffness}")
            optimizer.update_samples(samples[i], stiffness)
        else:
            next_sample = optimizer.get_next_sample()
            traj = find_valid_trajectory(robot, xyToRobotPose(next_sample, robot), robot.q)
            stiffness = execute_trajectory(robot, traj, box_plane, box_size, dt)
            print(f"Updating sample {i} with sample {next_sample}, stiffness {stiffness}")
            optimizer.update_samples(next_sample, stiffness)

    print("finish sampling")
    optimizer.plot_optimization()
