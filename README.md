# MedicalRoboticsProject

I only made this work in Ubuntu.

# Dependencies and how to build
Open terminal and type following commands
- cd Desktop
- mkdir MedRob
- code .

This should open VS code. From here you want to create a virtual environment by pressing ctrl+shift+p, and select python: create environment, and in the submenu select .venv
When the virtual environment is create you want to install the following dependencies into the virtual environment by first activating the environment with the command: source .venv/bin/activate

Now you should see the (.venv) in your terminal before your promt.

dependencies:
 - Roboticstoolbox-python: pip3 install roboticstoolbox-python
 -  pytransform3d: pip install 'pytransform3d[all]'
 -  websockets version 13.1: pip uninstall websockets -> pip install websockets==13.1
 -  numpy<2: (if numpy installed) pip uninstall numpy -y -> pip install numpy<2
 -  sdu_controllers: git clone https://github.com/SDU-Robotics/sdu_controllers.git
     - git submodule update --init --recursive
     - python -m pip install .

Now you should be able to run the python script called SimTest.py or SimpleSim.py to verify that eveerything builds and works.

# PalpationRobot.py
This file is the project, that implements the position admittance controller (complient controller) from the sdu_controllers library, and performs palpation on a phantom.
This simulation was made in the context of the course introduction to medical robotics at SDU, for a UR3 robot to perform palpation on a phantom -> later human, to detect cancerous anomalies in the soft tissue areas on humans.
