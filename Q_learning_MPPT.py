import numpy as np
import math
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import minimize
from math import log, exp, atan, pi

# --- PV parameters at STC ---
ISCR_STC = 71.73
VOC_STC = 366
V_MPPR = 293
I_MPPR = 67.23
niscT = 0.0010199
nvocT = -0.00361

# --- Parameters RL ---
WP = 1
WN = 4
I_BINS = 20
V_BINS = 20
DEG_THRESHOLD_DEGREES = 5
ACTION_SPACE_VALUES = [-0.1, -0.01, 0, 0.01, 0.1]
# ACTION_SPACE_VALUES = [-0.05, -0.02, -0.01, 0, 0.01, 0.02, 0.05]
M_MULTIPLIER = 4
RESPONSE_CYCLE_S = 0.01


# --- Physical model of PV ---
class PhysicalPVModel:
    def __init__(self):
        self.V_orc = VOC_STC
        self.I_scr = ISCR_STC
        self.V_mppr = V_MPPR
        self.I_mppr = I_MPPR
        self.niscT = niscT
        self.nvocT = nvocT
        self.Rl = 0.5
        self.T_pv = 25
        self.G_pv = 1000
        self.G_r = 1000
        self.T_r = 25
        
    def set_environment(self, irradiance, temperature):
        self.G_pv = irradiance
        self.T_pv = temperature

    def get_state(self, duty_cycle, prev_state):
        V_prev, I_prev = prev_state

    # Calculate model coefficients
        b_STC = log(1 - self.I_mppr / self.I_scr) / (self.V_mppr - self.V_orc)
        a1 = self.I_scr * exp(-b_STC * self.V_orc)
        b1 = b_STC / (1 + self.nvocT * (self.T_pv - self.T_r))
        I_sc = self.I_scr * self.G_pv / self.G_r * (1 + self.niscT * (self.T_pv - self.T_r))

    # Objective function to find I_pv
        def objective(I_pv):
            return (I_pv[0] - I_sc + a1 * exp(b1 * I_pv[0] * self.Rl / duty_cycle**2))**2

        result = minimize(objective, [0], method='BFGS')
        I_pv = result.x[0]
        V_pv = self.Rl / duty_cycle**2 * I_pv

    # Calculate change angle (Deg)
        if V_pv != V_prev:
            Deg = 180/pi * (atan(I_pv / V_pv) + atan((I_pv - I_prev) / (V_pv - V_prev)))
        else:
            Deg = 0

        return np.array([V_pv, I_pv, Deg])
    
    def get_output(self, D, prev_state=[280, 60]):
        V, I, _ = self.get_state(D, prev_state)
        P = V * I
        return I, V, P

    def estimate_mpp_power(self):
        return self.V_mppr * self.I_mppr * self.G_pv / self.G_r


# --- Controller RL MPPT ---
class RLMPPTController:
    def __init__(self, alpha=0.1, gamma=0.9):
        self.alpha = alpha
        self.gamma = gamma
        self.iscr_stc = ISCR_STC
        self.voc_stc = VOC_STC
        self.wp = WP
        self.wn = WN
        self.i_bins = I_BINS
        self.v_bins = V_BINS
        self.deg_threshold = math.radians(DEG_THRESHOLD_DEGREES)
        self.action_space = ACTION_SPACE_VALUES
        self.num_actions = len(self.action_space)
        self.Q_table = defaultdict(lambda: 0.0)
        self.exploration_counts = defaultdict(lambda: 0)
        self.num_exploration_rounds = self.num_actions * M_MULTIPLIER
        self.response_cycle_s = RESPONSE_CYCLE_S
        self.last_I = self.last_V = self.last_P = None
        self.last_state = self.last_action_idx = None
        self.current_duty_cycle = 0.5

    def _discretize_state(self, I, V, Deg):
        i_norm = np.clip(I / self.iscr_stc, 0, 1)
        v_norm = np.clip(V / self.voc_stc, 0, 1)
        i_idx = min(int(i_norm * self.i_bins), self.i_bins - 1)
        v_idx = min(int(v_norm * self.v_bins), self.v_bins - 1)
        deg_bin = 0 if abs(Deg) < self.deg_threshold else 1
        return (i_idx, v_idx, deg_bin)

    def _calculate_deg(self, i_new, v_new, i_old, v_old):
        epsilon = 1e-6
        delta_V = v_new - v_old
        delta_I = i_new - i_old
        if abs(delta_V) < epsilon:
            dIPV_dVPV = 0 if abs(delta_I) < epsilon else (float('inf') if delta_I > 0 else float('-inf'))
        else:
            dIPV_dVPV = delta_I / delta_V
        IPV_VPV = i_new / (v_new + epsilon)
        return math.atan(dIPV_dVPV) + math.atan(IPV_VPV)

    def _choose_action(self, state):
        if self.exploration_counts[state] < self.num_exploration_rounds:
            action_idx = random.randrange(self.num_actions)
            self.exploration_counts[state] += 1
        else:
            q_vals = [self.Q_table[(state, i)] for i in range(self.num_actions)]
            action_idx = int(np.argmax(q_vals))
        return action_idx

    def _calculate_reward(self, current_P, prev_P):
        delta = (current_P - prev_P) / self.response_cycle_s
        if abs(delta * self.response_cycle_s) < 0.001:
            return 0.0
        return self.wp * delta if delta >= 0 else self.wn * delta

    def _update_q_table(self, prev_state, action_idx, reward, next_state):
        max_q_next = max([self.Q_table[(next_state, i)] for i in range(self.num_actions)])
        current_q = self.Q_table[(prev_state, action_idx)]
        self.Q_table[(prev_state, action_idx)] = current_q + self.alpha * (reward + self.gamma * max_q_next - current_q)

    def control_step(self, I, V, P):
        if self.last_I is None or self.last_V is None:
            Deg = 0.0
            state = self._discretize_state(I, V, Deg)
            action_idx = self._choose_action(state)
        else:
            Deg = self._calculate_deg(I, V, self.last_I, self.last_V)
            state = self._discretize_state(I, V, Deg)
            reward = self._calculate_reward(P, self.last_P)
            self._update_q_table(self.last_state, self.last_action_idx, reward, state)
            action_idx = self._choose_action(state)

        delta_D = self.action_space[action_idx]
        self.current_duty_cycle = np.clip(self.current_duty_cycle + delta_D, 0.01, 0.99)

        self.last_I, self.last_V, self.last_P = I, V, P
        self.last_state = state
        self.last_action_idx = action_idx
        return self.current_duty_cycle



# --- Main simulation ---
if __name__ == "__main__":
    controller = RLMPPTController()
    pv = PhysicalPVModel()

# 1. Fix environmental conditions
    # pv.set_environment(irradiance=1000, temperature=25)

    # steps = 1000
    # D = controller.current_duty_cycle
    # duty_history = []
    # power_history = []
    # mpp_power_history = []

    # for step in range(steps):
    #     I, V, P = pv.get_output(D)
    #     mpp_power = pv.estimate_mpp_power()

    #     D = controller.control_step(I, V, P)

    #     duty_history.append(D)
    #     power_history.append(P)
    #     mpp_power_history.append(mpp_power)
        

    #     if step % 100 == 0:
    #         print(f"Step {step}: D={D:.3f}, P={P:.2f} W, MPP={mpp_power:.2f} W")

    # print("Training complete.")
    # print(f"Final power error: {abs(mpp_power_history[-1] - power_history[-1]):.2f} W")


# 2. Simulate changing temperature conditions
    # steps = 3000  # 3 stages, each 300 steps (3 seconds)

    # # --- Initialize the initial environment ---
    # pv.set_environment(irradiance=1000, temperature=25)

    # D = controller.current_duty_cycle
    # duty_history = []
    # power_history = []
    # mpp_power_history = []
    # temperature_history = []

    # for step in range(steps):
    #     # Change temperature according to each stage
    #     if step == 1000:
    #         pv.set_environment(irradiance=1000, temperature=40)
    #     elif step == 2000:
    #         pv.set_environment(irradiance=1000, temperature=60)

    #     I, V, P = pv.get_output(D)
    #     mpp_power = pv.estimate_mpp_power()

    #     D = controller.control_step(I, V, P)

    #     duty_history.append(D)
    #     power_history.append(P)
    #     mpp_power_history.append(mpp_power)
    #     temperature_history.append(pv.T_pv)

    # if step % 100 == 0:
    #     print(f"Step {step}: T={pv.T_pv}°C, D={D:.3f}, P={P:.2f} W, MPP={mpp_power:.2f} W")

    # print("Training complete.")
    # print(f"Final power error: {abs(mpp_power_history[-1] - power_history[-1]):.2f} W")


# 3. Simulate changing irradiance conditions
    steps = 3000  # 900 steps = 9 seconds

    # Stage 1: G = 400 W/m²
    # Stage 2: G = 1000 W/m²
    # Stage 3: G = 600 W/m²

    pv.set_environment(irradiance=1000, temperature=25)  # Initial assignment
    D = controller.current_duty_cycle
    duty_history = []
    power_history = []
    mpp_power_history = []
    irradiance_history = []

    for step in range(steps):
        if step == 1000:
            pv.set_environment(irradiance=600, temperature=25)
        elif step == 2000:
            pv.set_environment(irradiance=200, temperature=25)

        I, V, P = pv.get_output(D)
        mpp_power = pv.estimate_mpp_power()

        D = controller.control_step(I, V, P)

        duty_history.append(D)
        power_history.append(P)
        mpp_power_history.append(mpp_power)
        irradiance_history.append(pv.G_pv)  

        if step % 100 == 0:
            print(f"Step {step}: G={pv.G_pv} W/m², D={D:.3f}, P={P:.2f} W, MPP={mpp_power:.2f} W")



    # # --- Plot  ---
    # plt.figure(figsize=(12, 10))

# Plot 1. PV Power and MPP  
    # plt.subplot(3, 1, 1)
    # plt.plot(power_history, label="PV Power (RL)")
    # plt.plot(mpp_power_history, '--', label="Reference PV Power at Maximum Power Point (MPP)")
    # plt.ylabel("Power (W)")
    # plt.title("PV Power and Maximum Power Point (MPP)")
    # plt.legend()
    # plt.grid(True)

    # # 2. Duty Cycle
    # plt.subplot(3, 1, 2)
    # plt.plot(duty_history, label="Duty Cycle", color='green')
    # plt.ylabel("Duty Cycle")
    # plt.title("Duty Cycle during training")
    # plt.legend()
    # plt.grid(True)

    # # 3. Final power error: P - Pmax
    # plt.subplot(3, 1, 3)
    # plt.plot(np.array(power_history) - np.array(mpp_power_history), label="P - Pmax", color='red')
    # plt.axhline(0, linestyle='--', color='gray')
    # plt.xlabel("Simulation step")
    # plt.ylabel("P - Pmax (W)")
    # plt.title("Standard Test Conditions (STC) (T = 25°C, Ir = 1000 W/m²)")
    # plt.legend()
    # plt.grid(True)
    # --- Plot Temperature, PV Power, and Duty Cycle ---
    time_axis = np.arange(len(power_history)) * RESPONSE_CYCLE_S

    plt.figure(figsize=(12, 8))

# Plot 2: Temperature
    # plt.subplot(3, 1, 1)
    # plt.plot(time_axis, temperature_history, label="PV Temperature (°C)", color='orange')
    # plt.ylabel("Temperature (°C)")
    # plt.title("PV Temperature Variation")
    # plt.legend()
    # plt.grid(True)

    # # Plot 2: PV Power and MPP Reference
    # plt.subplot(3, 1, 2)
    # plt.plot(time_axis, power_history, label="PV Power (RL)", color='blue')
    # # plt.plot(time_axis, mpp_power_history, '--', label="Reference MPP Power", color='green')
    # plt.ylabel("Power (W)")
    # plt.title("PV Power Tracking")
    # plt.legend()
    # plt.grid(True)

    # # Plot 3: Duty Cycle
    # plt.subplot(3, 1, 3)
    # plt.plot(time_axis, duty_history, label="Duty Cycle", color='purple')
    # plt.xlabel("Time (s)")
    # plt.ylabel("Duty Cycle (D)")
    # plt.title("Duty Cycle Adjustment")
    # plt.legend()
    # plt.grid(True)


# Plot 3: Irradiance (G_pv)
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, irradiance_history, label="Irradiance (G_pv)", color='gold')
    plt.ylabel("Irradiance (W/m²)")
    plt.title("Irradiance Variation")
    plt.legend()
    plt.grid(True)

    # Plot 2: PV Power (RL)
    plt.subplot(3, 1, 2)
    plt.plot(time_axis, power_history, label="PV Power (RL)", color='blue')
    plt.ylabel("Power (W)")
    plt.title("PV Power Tracking")
    plt.legend()
    plt.grid(True)

    # Plot 3: Duty Cycle
    plt.subplot(3, 1, 3)
    plt.plot(time_axis, duty_history, label="Duty Cycle (D)", color='purple')
    plt.xlabel("Time (s)")
    plt.ylabel("Duty Cycle (D)")
    plt.title("Duty Cycle Adjustment")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


