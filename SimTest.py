import roboticstoolbox as rtb
import spatialgeometry as sg
from spatialmath import SE3, SO3
from roboticstoolbox.backends.swift import Swift
import time

robot = rtb.models.UR3()
robot.tool = SE3(0, 0, 0.06)
#tcp = sg.Sphere(radius=0.01, pose=robot.tool)
tcp = sg.Sphere(radius=0.01, pose=robot.fkine(robot.q) * robot.tool)  # Initialize sphere at TCP

env = Swift()
env.launch(realtime=True)
env.add(robot)
env.add(tcp)
env.step()

#time.sleep(10)

# Create a simulation environment
#robot.plot(robot.qz, block=True)