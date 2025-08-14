# Q-Learning & Deep Q-Learning (DQN) for MPPT in Photovoltaic (PV) Systems

This repository provides two reinforcement learning-based Maximum Power Point Tracking (MPPT) controllers for photovoltaic (PV) systems:
1. **Q-Learning MPPT** – Tabular Q-learning without prior PV curve knowledge.
2. **DQN MPPT** – Deep Q-Learning with neural network function approximation for better scalability.

Both controllers adjust the **duty cycle** of a DC/DC converter to maximize PV output power under varying irradiance and temperature conditions.

---

## Features

### Common Features
- Physical PV model based on Standard Test Conditions (STC).
- Adaptive MPPT control under:
  - Fixed environmental conditions.
  - Changing temperature.
  - Changing irradiance.
- Real-time duty cycle adjustment.
- Visualization of:
  - Irradiance & Temperature.
  - PV Power Tracking (Actual Power vs Maximum Power).
  - Duty Cycle Evolution.

### Q-Learning Specific
- Tabular Q-value updates.
- Discrete state-action representation.
- Suitable for small, well-defined state spaces.

### DQN Specific
- Neural network-based Q-function approximation.
- Replay buffer for experience storage.
- Mini-batch gradient descent training.
- Scales to larger, continuous state spaces.

---

## Requirements

Python **3.8+** is recommended.  
Install required dependencies:

```bash
pip install numpy matplotlib torch
