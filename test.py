import roboticstoolbox as rtb
from spatialmath import SE3, SO3
import spatialgeometry as sg
import numpy as np
import argparse
import sys

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="UR trajectory demo")
parser.add_argument("-b", "--backend", dest="backend", default="pyplot",
                    help="choose backend: pyplot (default), swift, vpython")
parser.add_argument("-m", "--model", dest="model", default="DH",
                    help="choose model: DH (default), URDF")
args = parser.parse_args()

# pick robot
if args.model.lower() == "dh":
    robot = rtb.models.DH.UR3()
elif args.model.lower() == "urdf":
    robot = rtb.models.UR3()
else:
    raise ValueError("unknown model")

# trajectory
qt = rtb.tools.trajectory.jtraj(robot.qz, robot.qr, 200)

if args.backend.lower() == "pyplot":
    if args.model.lower() != "dh":
        print("PyPlot only supports DH models for now")
        sys.exit(1)

    # 1) Define your base plane (1m×1m×1mm) and put it at z=0.2 m:
    T_plane = SE3.Rt(
        SO3.Rx(np.pi/12 * np.sin(0/2)),   # any tilt if you want
        [0.5, 0, 0.2]                     # centered at x=0.5, y=0, z=0.2
    )
    baseplane = sg.Cuboid([1, 1, 0.001], pose=T_plane)

    # 2) Define your block (0.1m cube) so its bottom face sits on top of the plane:
    block_height = 0.1
    # plane top is at z = 0.2 + 0.001/2 ≃ 0.2005; we’ll ignore the 0.5 mm half-thickness 
    # and just place cube bottom at z=0.2:
    z_bottom = 0.2
    z_center = z_bottom + block_height/2
    T_block = SE3(0.5, 0, z_center)      # same x,y as plane center
    block = sg.Cuboid([0.1, 0.1, block_height], pose=T_block)

    # 3) Plot robot (non-blocking) and grab its axes
    robot.plot(qt.q, backend="pyplot", block=False,
               limits=[-0.2, 0.8, -0.2, 0.8, 0.0, 1.0])
    ax = plt.gcf().axes[0]  # the 3D Axes created by RTB

    # 4) Plot the plane and block into that same axes
    baseplane.plot(ax=ax, facecolor="lightgray", edgecolor="k", alpha=0.5)
    block.plot(ax=ax,    facecolor="orange",    edgecolor="r", alpha=0.8)

    # 5) Tidy up
    ax.set_box_aspect((1,1,1))
    ax.view_init(elev=30, azim=-45)
    plt.show()

else:
    # other backends: just do the normal plot call
    robot.plot(qt.q, backend=args.backend)
