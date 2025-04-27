import asyncio

import roboticstoolbox as rtb
import roboticstoolbox.backends.swift as rtb_swift
from spatialmath import SE3
from math import pi

async def main():
    # 1) Create Swift and launch it *inside* the running asyncio loop
    env = rtb_swift.Swift()
    env.launch()       # now websockets.serve() finds a running loop

    # 2) Load the UR3 model and add it
    robot = rtb.models.URDF.UR3()
    env.add(robot)

    # 3) Set an initial joint configuration
    robot.q = [0, -pi/4, -pi/4, -pi/2, 0, 0]

    # 4) Compute initial and target Cartesian poses
    initial_pose = robot.fkine()
    target_pose  = initial_pose * SE3.Tx(0.20)  # +20 cm in X

    # 5) Move in a straight line over 50 steps
    steps = 50
    for i in range(steps + 1):
        frac = i / steps
        # pure translation interpolation
        interp_pose = initial_pose * SE3.Tx(0.20 * frac)

        # inverse kinematics (Levenberg–Marquardt)
        sol = robot.ikine_LM(interp_pose)
        if not sol.success:
            print(f"IK failed at step {i}: {sol.reason}")
            break

        robot.q = sol.q              # update joints
        env.step(0.05)               # advance Swift by 50 ms
        await asyncio.sleep(0.05)    # yield to the loop so the WebSocket server can run

    # 6) Keep window open until user closes
    print("Motion done. Press Ctrl+C to exit.")
    while True:
        await asyncio.sleep(1.0)

if __name__ == "__main__":
    asyncio.run(main())
