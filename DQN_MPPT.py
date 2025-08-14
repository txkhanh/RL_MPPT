# Deep Q-Learning version of the original RL MPPT controller
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque
from scipy.optimize import minimize
from math import log, exp, atan, pi

# --- Constants ---
ISCR_STC = 71.73
VOC_STC = 366
V_MPPR = 293
I_MPPR = 67.23
niscT = 0.0010199
nvocT = -0.00361
# ACTION_SPACE_VALUES = [-0.1, -0.01, 0, 0.01, 0.1]
ACTION_SPACE_VALUES = [-0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02]

RESPONSE_CYCLE_S = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PV model ---
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
        b_STC = log(1 - self.I_mppr / self.I_scr) / (self.V_mppr - self.V_orc)
        a1 = self.I_scr * exp(-b_STC * self.V_orc)
        b1 = b_STC / (1 + self.nvocT * (self.T_pv - self.T_r))
        I_sc = self.I_scr * self.G_pv / self.G_r * (1 + self.niscT * (self.T_pv - self.T_r))

        def objective(I_pv):
            return (I_pv[0] - I_sc + a1 * exp(b1 * I_pv[0] * self.Rl / duty_cycle**2))**2

        result = minimize(objective, [0], method='BFGS')
        I_pv = result.x[0]
        V_pv = self.Rl / duty_cycle**2 * I_pv

        Deg = 180/pi * (atan(I_pv / V_pv) + atan((I_pv - I_prev) / (V_pv - V_prev))) if V_pv != V_prev else 0
        return np.array([V_pv, I_pv, Deg])

    def get_output(self, D, prev_state=[280, 60]):
        V, I, _ = self.get_state(D, prev_state)
        return I, V, V * I

    def estimate_mpp_power(self):
        return self.V_mppr * self.I_mppr * self.G_pv / self.G_r

# --- DQN Network ---
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# --- DQN Controller ---
class DQNMPPTController:
    def __init__(self, state_dim=3, action_space=ACTION_SPACE_VALUES):
        self.state_dim = state_dim
        self.action_space = action_space
        self.num_actions = len(action_space)
        self.model = DQN(state_dim, self.num_actions).to(device)
        self.target_model = DQN(state_dim, self.num_actions).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.memory = deque(maxlen=5000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.995
        self.update_target_every = 50
        self.step_count = 0
        self.current_duty_cycle = 0.5
        self.last_state = None
        self.last_action = None

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_vals = self.model(state_tensor)
        return torch.argmax(q_vals).item()

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)

        q_values = self.model(states).gather(1, actions)
        next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
        expected_q = rewards + self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.step_count % self.update_target_every == 0:
            self.target_model.load_state_dict(self.model.state_dict())


    def control_step(self, state, reward):
        action_idx = self.act(state)
        delta_D = self.action_space[action_idx]
        self.current_duty_cycle = np.clip(self.current_duty_cycle + delta_D, 0.01, 0.99)
        if self.last_state is not None and self.last_action is not None:
            self.remember(self.last_state, self.last_action, reward, state)
            
        self.replay()
        self.last_state = state
        self.last_action = action_idx
        self.step_count += 1
        return self.current_duty_cycle

# --- Main Simulation ---
if __name__ == "__main__":
    controller = DQNMPPTController()
    pv = PhysicalPVModel()
    pv.set_environment(irradiance=1000, temperature=25)

    steps = 1000
    duty_history, power_history, mpp_power_history = [], [], []
    prev_state = [280, 60]

    for step in range(steps):
        I, V, P = pv.get_output(controller.current_duty_cycle, prev_state)
        prev_state = [V, I]
        mpp_power = pv.estimate_mpp_power()
        Deg = 180/pi * (atan(I / V)) if V != 0 else 0
        state = np.array([V / VOC_STC, I / ISCR_STC, Deg / 180])
        reward = (P - mpp_power) / mpp_power
        D = controller.control_step(state, reward)

        duty_history.append(D)
        power_history.append(P)
        mpp_power_history.append(mpp_power)

        if step % 100 == 0:
            print(f"Step {step}: D={D:.3f}, P={P:.2f} W, MPP={mpp_power:.2f} W")

    print("Training completed.")

    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    plt.plot(power_history, label="PV Power (DQN)")
    plt.plot(mpp_power_history, '--', label="Reference MPP Power")
    plt.ylabel("Power (W)")
    plt.title("PV Power and MPP")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(duty_history, label="Duty Cycle", color='green')
    plt.ylabel("Duty Cycle")
    plt.title("Duty Cycle over Time")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(np.array(power_history) - np.array(mpp_power_history), label="P - Pmax", color='red')
    plt.axhline(0, linestyle='--', color='gray')
    plt.xlabel("Simulation Step")
    plt.ylabel("Power Error (W)")
    plt.title("Power Tracking Error")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
