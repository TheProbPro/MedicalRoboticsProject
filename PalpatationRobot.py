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

_logger = logging.getLogger('Palpation_robot')

if __name__ == '__main__':
    # Setup robot parameters
    freq = 500
    dt = 1/freq

    # Initialize robot
    robot = rtb.models.UR3()
    q_start = np.array([0, -np.pi / 4, np.pi / 4, -np.pi / 4, np.pi / 2, 0])
    robot.q = q_start

    # Setup environment
    env = Swift()

    # define floor/table plane
    floor_plane = sg.Cuboid([10,10,0.01], pose=robot.base, collision=True)

    # define the place the cube should be on
    box_plane = SE3(0.3,0,0)

    # Create box
    box = sg.Cuboid([0.1,0.1,0.1], pose=box_plane, collision=True)

    # Add robot and box to environment
    env.launch(frequency=freq, realtime=True)
    env.add(robot, collision_alpha=1.0)
    env.add(box, collision_alpha=1.0)
    env.add(floor_plane, collision_alpha=0.1)
    env.step()

    time.sleep(1)

    #define target pose 0.2 above the box and tool oriented to point at the box
    orientation = SE3.Rt(SO3.Ry(np.pi/2), [0, 0, 0.1])
    target_pose = box_plane * orientation#SE3.Trans(0, 0, 0.2) * SO3.rpy(0, np.pi/2, np.pi/2)#SE3.Rx(-np.pi / 2) * SE3.Ry(np.pi / 2) * SE3.Rz(np.pi / 2)

    # Check if any joints go below the floor_plane
    while True:
        # Plan trajectory
        ik_solution = robot.ikine_LM(target_pose)

        if not ik_solution.success:
            raise RuntimeError("IK failed to find a solution.")
    
        q_target = ik_solution.q
        trajectory = rtb.jtraj(q_start, q_target, 1000)

        # Check if any joint goes below z=0
        valid_trajectory = True
        for q in trajectory.q:
            fk = robot.fkine_all(q)  # Compute forward kinematics
            for joint_pose in fk:
                if joint_pose.t[2] < 0:  # Check if z-coordinate is below 0
                    valid_trajectory = False
                    print("Joint below z=0 detected!")
                    break
            
            # Check for singularity
            J = robot.jacob0(q)  # Compute the Jacobian at the current configuration
            if np.linalg.det(J) == 0:  # High condition number indicates a singularity
                valid_trajectory = False
                print("Singularity detected!")
                break

        if valid_trajectory:
            break  # Exit loop if trajectory is valid
        #else:
        #   # Adjust target pose slightly upward and retry
        #    target_pose = target_pose * SE3.Trans(0, 0, 0.001)

    # Execute the valid trajectory
    for q in trajectory.q:
        robot.q = q
        env.step(dt)

    print('Moved to box')
