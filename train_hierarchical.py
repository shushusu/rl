import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import random

from hierarchical_env import HierarchicalESSEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

EXPERIMENT_TAG_BASE = "a2_reward_tuned"
SEEDS = [0, 1, 2]


def set_seed(seed: int):
    print(f"[Seed] Using seed = {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return x


class TransformerPPO(nn.Module):
    def __init__(self, obs_dim, act_dim, d_model=128, nhead=4, num_layers=2, seq_len=24):
        super(TransformerPPO, self).__init__()

        self.embedding = nn.Linear(obs_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.actor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
            nn.Tanh(),
        )

        self.critic = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return x[:, -1, :]  

    def get_action(self, x):
        features = self.forward(x)
        mean = self.actor(features)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

class HistoryBuffer:
    def __init__(self, num_agents, obs_dim, seq_len):
        self.buffer = np.zeros((num_agents, seq_len, obs_dim), dtype=np.float32)

    def push(self, new_obs):
        self.buffer[:, :-1, :] = self.buffer[:, 1:, :]
        self.buffer[:, -1, :] = new_obs

    def get(self):
        return self.buffer

def update_ppo(agent, optimizer, memory, batch_size=4096, epochs=10, gamma=0.99, clip_ratio=0.2):
    if not memory:
        return

    states_list, actions_list, logprobs_list, rewards_list, dones_list = zip(*memory)

    b_states = torch.FloatTensor(np.concatenate(states_list, axis=0)).to(device)
    b_actions = torch.cat(actions_list).to(device)
    b_logprobs = torch.cat(logprobs_list).to(device)

    np_rewards = np.array(rewards_list)
    if np_rewards.ndim == 1:
        np_rewards = np_rewards.reshape(-1, 1)

    np_dones = np.array(dones_list)
    if np_dones.ndim == 1:
        np_dones = np_dones.reshape(-1, 1)

    R = np.zeros(np_rewards.shape[1], dtype=np.float32)
    batch_returns = np.zeros_like(np_rewards)
    for t in reversed(range(len(np_rewards))):
        R = np_rewards[t] + gamma * R * (1 - np_dones[t])
        batch_returns[t] = R

    b_returns = torch.FloatTensor(batch_returns.flatten()).to(device)
    b_returns = (b_returns - b_returns.mean()) / (b_returns.std() + 1e-8)

    dataset_size = b_states.size(0)
    real_agent = agent  

    for _ in range(epochs):
        indices = torch.randperm(dataset_size)

        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            idx = indices[start:end]

            mini_states = b_states[idx]
            mini_actions = b_actions[idx]
            mini_logprobs = b_logprobs[idx]
            mini_returns = b_returns[idx]

            features = agent(mini_states)

            mean = real_agent.actor(features)
            std = torch.exp(real_agent.log_std)
            dist = torch.distributions.Normal(mean, std)

            new_logprobs = dist.log_prob(mini_actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            state_values = real_agent.critic(features)

            ratio = torch.exp(new_logprobs - mini_logprobs)
            advantage = mini_returns - state_values.squeeze()

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_values.squeeze(), mini_returns)
            entropy_loss = -0.01 * entropy.mean()

            loss = actor_loss + 0.5 * critic_loss + entropy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.cuda.empty_cache()

def train_for_seed(seed: int):
    set_seed(seed)

    env = HierarchicalESSEnv("ma_train_data.npy")

    SEQ_LEN = 24
    LR = 1e-4
    MAX_EPISODES = 50000
    UPDATE_INTERVAL_EPISODES = 20

    tag = f"{EXPERIMENT_TAG_BASE}_seed{seed}"
    print(f"\n==============================")
    print(f"[Experiment] TAG = {tag}")
    print(f"==============================")

    manager = TransformerPPO(env.mgr_obs_dim, env.mgr_act_dim, seq_len=SEQ_LEN, d_model=128, nhead=4, num_layers=2).to(device)
    worker = TransformerPPO(env.wrk_obs_dim, env.wrk_act_dim, seq_len=SEQ_LEN, d_model=128, nhead=4, num_layers=2).to(device)

    mgr_opt = optim.Adam(manager.parameters(), lr=LR)
    wrk_opt = optim.Adam(worker.parameters(), lr=LR)

    total_updates = MAX_EPISODES // UPDATE_INTERVAL_EPISODES
    mgr_sched = optim.lr_scheduler.LinearLR(mgr_opt, start_factor=1.0, end_factor=0.01, total_iters=total_updates)
    wrk_sched = optim.lr_scheduler.LinearLR(wrk_opt, start_factor=1.0, end_factor=0.01, total_iters=total_updates)

    mgr_buf = HistoryBuffer(1, env.mgr_obs_dim, SEQ_LEN)
    wrk_buf = HistoryBuffer(env.n_agents, env.wrk_obs_dim, SEQ_LEN)

    mgr_mem = []
    wrk_mem = []

    history_mgr_r = []
    history_wrk_r = []
    history_avg_cost = []
    history_peak_ratio = []

    for ep in range(1, MAX_EPISODES + 1):
        mgr_obs, wrk_obs = env.reset()

        mgr_buf.buffer[:] = mgr_obs.reshape(1, 1, -1)
        wrk_buf.buffer[:] = wrk_obs[:, None, :]

        ep_mgr_reward = 0.0
        ep_wrk_reward = 0.0
        ep_total_cost = 0.0
        ep_over_count = 0

        while True:
            mgr_seq = torch.FloatTensor(mgr_buf.get()).to(device)
            wrk_seq = torch.FloatTensor(wrk_buf.get()).to(device)

            with torch.no_grad():
                raw_mgr_act, mgr_log = manager.get_action(mgr_seq)
                wrk_act, wrk_log = worker.get_action(wrk_seq)

                mgr_act_val = (torch.tanh(raw_mgr_act) + 1.0) / 2.0

            mgr_np = mgr_act_val.cpu().numpy().flatten()
            wrk_np = wrk_act.cpu().numpy()

            (next_mgr, next_wrk), (r_m, r_w), done, info = env.step(mgr_np, wrk_np)

            mgr_mem.append(
                (mgr_buf.get().copy(),
                 raw_mgr_act.detach().cpu(),
                 mgr_log.detach().cpu(),
                 r_m,
                 done)
            )
            wrk_mem.append(
                (wrk_buf.get().copy(),
                 wrk_act.detach().cpu(),
                 wrk_log.detach().cpu(),
                 r_w,
                 done)
            )

            mgr_buf.push(next_mgr)
            wrk_buf.push(next_wrk)

            ep_mgr_reward += r_m
            ep_wrk_reward += np.mean(r_w)
            ep_total_cost += info.get("total_cost", 0.0)
            if abs(info.get("total_load", 0.0)) > env.PEAK_THRESHOLD:
                ep_over_count += 1

            if done:
                break

        if ep % UPDATE_INTERVAL_EPISODES == 0:
            update_ppo(manager, mgr_opt, mgr_mem, batch_size=4096, epochs=10)
            update_ppo(worker, wrk_opt, wrk_mem, batch_size=4096, epochs=10)

            mgr_sched.step()
            wrk_sched.step()

            mgr_mem = []
            wrk_mem = []

        history_mgr_r.append(ep_mgr_reward)
        history_wrk_r.append(ep_wrk_reward)
        avg_cost_ep = ep_total_cost / env.MAX_EPISODE_STEPS
        peak_ratio_ep = ep_over_count / env.MAX_EPISODE_STEPS
        history_avg_cost.append(avg_cost_ep)
        history_peak_ratio.append(peak_ratio_ep)

        if ep % 100 == 0:
            avg_m = np.mean(history_mgr_r[-100:])
            avg_w = np.mean(history_wrk_r[-100:])
            avg_c = np.mean(history_avg_cost[-100:])
            avg_p = np.mean(history_peak_ratio[-100:])
            current_lr = mgr_opt.param_groups[0]["lr"]
            print(
                f"Ep {ep}/{MAX_EPISODES} | "
                f"Mgr: {avg_m:.2f} | Wrk: {avg_w:.2f} | "
                f"AvgCost: {avg_c:.2f} | PeakRatio: {avg_p:.3f} | LR: {current_lr:.2e}"
            )

    torch.save(manager.state_dict(), f"manager_transformer_{tag}.pth")
    torch.save(worker.state_dict(), f"worker_transformer_{tag}.pth")

    plt.figure(figsize=(10, 5))
    plt.plot(history_mgr_r, label="Manager Reward", alpha=0.6)
    plt.plot(history_wrk_r, label="Worker Reward", alpha=0.6)
    plt.legend()
    plt.title(f"H-MARL Training Result ({tag})")
    plt.savefig(f"training_result_{tag}.png")
    plt.close()
    print(f"Training complete for seed={seed} | TAG={tag}")

def main():
    for seed in SEEDS:
        train_for_seed(seed)

if __name__ == "__main__":
    main()