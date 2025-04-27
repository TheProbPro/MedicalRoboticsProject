import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Define a helper function to draw a box (cuboid) in the scene.
def draw_box(ax, center, size):
    """
    Draw a box given an axis, center, and size.
    
    Parameters:
        ax (matplotlib.axes._subplots.Axes3DSubplot): The 3D axis.
        center (array-like): The (x, y, z) coordinates of the box center.
        size (float): The edge length of the cube.
    """
    r = size / 2.0
    # Compute the 8 vertices of the cube.
    corners = np.array([
        [center[0]-r, center[1]-r, center[2]-r],
        [center[0]+r, center[1]-r, center[2]-r],
        [center[0]+r, center[1]+r, center[2]-r],
        [center[0]-r, center[1]+r, center[2]-r],
        [center[0]-r, center[1]-r, center[2]+r],
        [center[0]+r, center[1]-r, center[2]+r],
        [center[0]+r, center[1]+r, center[2]+r],
        [center[0]-r, center[1]+r, center[2]+r]
    ])
    # Define the 6 faces of the cube by connecting appropriate vertices.
    faces = [
        [corners[j] for j in [0, 1, 2, 3]],  # bottom face
        [corners[j] for j in [4, 5, 6, 7]],  # top face
        [corners[j] for j in [0, 1, 5, 4]],  # side face
        [corners[j] for j in [2, 3, 7, 6]],  # opposite side
        [corners[j] for j in [1, 2, 6, 5]],  # front face
        [corners[j] for j in [4, 7, 3, 0]]   # back face
    ]
    # Create a translucent 3D polygon collection to represent the box.
    box = Poly3DCollection(faces, alpha=0.3, edgecolor='k')
    ax.add_collection3d(box)

# 1. Create the UR3 Robot Model
# The Robotics Toolbox for Python provides models for many robots.
# Here, we assume that the UR3 model is available as rtb.models.UR3().
robot = rtb.models.UR3()

# 2. Set an initial joint configuration (in radians).
# The configuration below is just one example and may be adjusted as needed.
q0 = np.radians([0, -90, 90, -90, -90, 0])

# 3. Create a matplotlib figure and axis for plotting.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 4. Plot the robot in its initial configuration.
# Using block=False allows us to update the same figure later.
robot.plot(q0, ax=ax, block=False)

# 5. Compute the initial end-effector pose using forward kinematics.
T_start = robot.fkine(q0)

# 6. Define a target end-effector pose.
# Here, we translate the starting pose by 0.2 m along the x-direction.
dx = 0.2
T_final = T_start * SE3.Trans(dx, 0, 0)

# 7. Draw a box at the target location.
# The box is centered at the target translation (T_final.t) with an edge length of 0.1 m.
box_center = T_final.t  # Extract the translation vector from the transform
draw_box(ax, box_center, 0.1)
plt.draw()

# 8. Create a linear trajectory for the end-effector using interpolation.
steps = 50
s_vals = np.linspace(0, 1, steps)
traj = [T_start.interp(T_final, s) for s in s_vals]

# 9. Compute corresponding joint configurations along the trajectory.
# We use an iterative inverse kinematics solver (Levenberg-Marquardt method).
qs = []      # list to hold joint configurations
q_prev = q0  # start with the initial configuration as the guess

for T in traj:
    sol = robot.ikine_LM(T, q0=q_prev)
    qs.append(sol.q)
    q_prev = sol.q  # update the initial guess for the next iteration

# 10. Animate the robot following the trajectory.
for q in qs:
    robot.plot(q, ax=ax, block=False)
    plt.pause(0.05)  # pause briefly for the animation effect

plt.show()
