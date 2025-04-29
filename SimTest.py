import roboticstoolbox as rtb

robot = rtb.models.UR3()

# Create a simulation environment
robot.plot(robot.qz, block=True)