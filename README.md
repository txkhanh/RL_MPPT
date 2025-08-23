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

1. Fix environmental conditions
<img width="3199" height="1774" alt="image" src="https://github.com/user-attachments/assets/c736402c-7e05-424c-be27-d918a4169ee3" />

2. Simulate changing temperature conditions
<img width="3199" height="1764" alt="image" src="https://github.com/user-attachments/assets/4dce8ba5-0b54-41c6-bf74-6a39846067ab" />

3. Simulate changing irradiance conditions
<img width="3190" height="1773" alt="image" src="https://github.com/user-attachments/assets/8ddba68c-f1a1-4f69-a1fb-9c2265f9d1eb" />

### DQN Specific
- Neural network-based Q-function approximation.
- Replay buffer for experience storage.
- Mini-batch gradient descent training.
- Scales to larger, continuous state spaces.

<img width="3199" height="1786" alt="image" src="https://github.com/user-attachments/assets/f918ee7c-8e09-45e5-aeb8-0e24cce47b7e" />


---

## Requirements

Python **3.8+** is recommended.  
Install required dependencies:

```bash
pip install numpy matplotlib torch
