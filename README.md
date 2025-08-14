# Q-Learning MPPT for Photovoltaic (PV) Systems

This repository provides an implementation of a **Q-learning-based Maximum Power Point Tracking (MPPT)** controller for photovoltaic (PV) systems. The controller adjusts the duty cycle of a DC/DC converter to maximize PV output power under varying irradiance and temperature conditions.

## Features
- **Physical PV model** based on Standard Test Conditions (STC).
- **Q-learning-based MPPT** without prior PV curve knowledge.
- Simulations under:
  - Fixed environmental conditions.
  - Changing temperature.
  - Changing irradiance.
- Real-time duty cycle adjustment.
- Visualization of irradiance/temperature, PV power tracking, and duty cycle.

## Requirements
Python **3.8+** is recommended. Install dependencies:
```bash
pip install numpy matplotlib scipy
