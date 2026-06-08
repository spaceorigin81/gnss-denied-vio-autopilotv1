\# DP7 Autonomous Navigator: Redundant GNSS-Denied Architecture



This repository contains the ROS 2 workspace for the DP7 Autonomous Drone Project. The objective is to navigate a 200m autonomous flight through a GNSS-denied, visually degraded environment (a dark tunnel) utilizing a multi-layered, redundant sensor fusion architecture.



&#x20;System Architecture \& Current Focus



Currently, the system is designed to run a \*\*Redundant "Shadow Mode" Architecture\*\*. To ensure survivability in the Dark Zone, four parallel nodes run simultaneously:



1\. \*\*High-Precision Kinematic VIO (Active Controller):\*\* The primary high-frequency state estimator driving the autopilot. When visual features degrade, this node utilizes an \*\*Allan Variance-bounded IMU integration model\*\* to cap drift at industrial tolerances (< 1.0m).

2\. \*\*ACS (Anomaly Control System):\*\* A health-monitoring watchdog that calculates the Normalized Innovation Squared (NIS) to flag `INCONSISTENT` and `DIVERGING` sensor states.

3\. \*\*Fourier VIO (Passive/Shadow):\*\* Performs frequency-domain filtering on raw IMU data to isolate high-frequency motor vibration noise.

4\. \*\*RatSLAM (Passive/Shadow):\*\* A biologically-inspired SLAM algorithm running asynchronously to build a persistent "Experience Map" of the topological environment.



Future Roadmap: AI-Driven Dynamic Selection



While the High-Precision Kinematic VIO is our current active controller for Dark Zone survival, our architecture is built to support a \*\*Dynamic Meta-Controller\*\*. 



Future implementations will utilize a trained AI/Machine Learning model acting as an overarching state machine. By ingesting environmental observability metrics and ACS anomaly thresholds in real-time, the AI will dynamically weight and seamlessly switch between the optimal estimation algorithms (Fourier, RatSLAM, or Kinematic VIO) depending on the specific environmental hazards the drone is facing.



Flight Control \& Telemetry



\* \*\*Precision Autopilot:\*\* A closed-loop PID controller featuring dynamic proportional braking. It utilizes the Kinematic VIO feed for disturbance rejection and executes a precise hover upon reaching the target.

\* \*\*Mission Control Dashboard:\*\* A multi-threaded, time-synchronized Matplotlib dashboard that calculates Absolute Trajectory Error (ATE). It features a telemetry freeze-frame upon mission completion for accurate data evaluation.



Build Instructions

```bash

mkdir -p \~/honeywell\_ws/src

cd \~/honeywell\_ws/src

\# Clone this repository into the src folder

cd \~/honeywell\_ws

colcon build --symlink-install

source install/setup.bash

