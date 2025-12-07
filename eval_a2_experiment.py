import numpy as np
import torch
import torch.nn as nn
import math

from hierarchical_env import HierarchicalESSEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

SEQ_LEN = 24 
MODEL_TAG = "reward_tuned"   

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerPPO(nn.Module):
    def __init__(self, obs_dim, act_dim, d_model=128, nhead=4, num_layers=2, seq_len=24):
        super().__init__()
        
        self.embedding = nn.Linear(obs_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=256, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.actor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim),
            nn.Tanh()
        )
        
        self.critic = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
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
    def __init__(self, num_agents, obs_dim, seq_len=24):
        self.buffer = np.zeros((num_agents, seq_len, obs_dim), dtype=np.float32)
        
    def push(self, new_obs):
        self.buffer[:, :-1, :] = self.buffer[:, 1:, :]
        self.buffer[:, -1, :] = new_obs
        
    def get(self):
        return self.buffer

def policy_no_ess(mgr_obs, wrk_obs, t, env, mgr_buf=None, wrk_buf=None, models=None):
    mgr_action = np.array([1.0], dtype=np.float32)
    wrk_actions = np.zeros((env.n_agents, 1), dtype=np.float32)
    return mgr_action, wrk_actions


def policy_rule_based(mgr_obs, wrk_obs, t, env, mgr_buf=None, wrk_buf=None, models=None):
    hours = t % 24
    n_agents = env.n_agents

    mgr_action = np.array([1.0], dtype=np.float32)  
    wrk_actions = np.zeros((n_agents, 1), dtype=np.float32)

    soc  = wrk_obs[:, 0]
    gen  = wrk_obs[:, 1]
    load = wrk_obs[:, 2]

    for i in range(n_agents):
        if 10 <= hours < 16:  
            if gen[i] > load[i] and soc[i] < 0.9:
                wrk_actions[i, 0] = +0.7
        elif 18 <= hours < 22:  
            if soc[i] > 0.3:
                wrk_actions[i, 0] = -0.7
        else:
            wrk_actions[i, 0] = 0.0

    return mgr_action, wrk_actions


def policy_htrans(mgr_obs, wrk_obs, t, env, mgr_buf, wrk_buf, models):
    manager, worker = models["manager"], models["worker"]

    mgr_seq = torch.FloatTensor(mgr_buf.get()).to(device)   
    wrk_seq = torch.FloatTensor(wrk_buf.get()).to(device)   

    with torch.no_grad():
        raw_mgr_act, _ = manager.get_action(mgr_seq)
        wrk_act, _     = worker.get_action(wrk_seq)

        mgr_act_val = (torch.tanh(raw_mgr_act) + 1.0) / 2.0

    mgr_np = mgr_act_val.cpu().numpy().flatten()
    wrk_np = wrk_act.cpu().numpy()
    return mgr_np, wrk_np

def evaluate_policy(env, policy_fn, episodes=30, use_htrans=False, models=None):
    results = []

    for ep in range(episodes):
        mgr_obs, wrk_obs = env.reset()
        ep_cost = 0.0
        over_count = 0
        max_margin = 0.0

        if use_htrans:
            mgr_buf = HistoryBuffer(1, env.mgr_obs_dim, SEQ_LEN)
            wrk_buf = HistoryBuffer(env.n_agents, env.wrk_obs_dim, SEQ_LEN)
            mgr_buf.buffer[:] = mgr_obs.reshape(1, 1, -1)
            wrk_buf.buffer[:] = wrk_obs[:, None, :]
        else:
            mgr_buf = None
            wrk_buf = None

        for t in range(env.MAX_EPISODE_STEPS):
            if use_htrans:
                mgr_action, wrk_actions = policy_fn(
                    mgr_obs, wrk_obs, t, env, mgr_buf, wrk_buf, models
                )
            else:
                mgr_action, wrk_actions = policy_fn(
                    mgr_obs, wrk_obs, t, env, mgr_buf, wrk_buf, models
                )

            (next_mgr, next_wrk), (_, _), done, info = env.step(mgr_action, wrk_actions)

            total_load = info["total_load"]
            total_cost = info["total_cost"] 

            ep_cost += total_cost

            if abs(total_load) > env.PEAK_THRESHOLD:
                over_count += 1
                margin = abs(total_load) - env.PEAK_THRESHOLD
                if margin > max_margin:
                    max_margin = margin

            mgr_obs, wrk_obs = next_mgr, next_wrk

            if use_htrans:
                mgr_buf.push(next_mgr.reshape(1, -1))
                wrk_buf.push(next_wrk)

            if done:
                break

        peak_ratio = over_count / env.MAX_EPISODE_STEPS
        results.append((ep_cost, peak_ratio, max_margin))

    arr = np.array(results)  
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)

    metrics = {
        "TotalCost_mean": float(mean[0]),
        "TotalCost_std":  float(std[0]),
        "PeakRatio_mean": float(mean[1]),
        "PeakRatio_std":  float(std[1]),
        "MaxMargin_mean": float(mean[2]),
        "MaxMargin_std":  float(std[2]),
    }
    return metrics

def main():
    env = HierarchicalESSEnv(data_path="ma_train_data.npy", is_train=True)
    print("=== No-ESS Baseline ===")
    metrics_no = evaluate_policy(env, policy_no_ess, episodes=30, use_htrans=False)
    print(metrics_no)

    print("\n=== Rule-based ESS ===")
    metrics_rule = evaluate_policy(env, policy_rule_based, episodes=30, use_htrans=False)
    print(metrics_rule)

    print("\n=== Learned H-Trans ===")
    manager = TransformerPPO(env.mgr_obs_dim, env.mgr_act_dim, seq_len=SEQ_LEN).to(device)
    worker  = TransformerPPO(env.wrk_obs_dim, env.wrk_act_dim, seq_len=SEQ_LEN).to(device)

    manager.load_state_dict(
        torch.load(f"manager_transformer_{MODEL_TAG}.pth", map_location=device, weights_only=False)
    )
    worker.load_state_dict(
        torch.load(f"worker_transformer_{MODEL_TAG}.pth", map_location=device, weights_only=False)
    )
    manager.eval()
    worker.eval()

    models = {"manager": manager, "worker": worker}
    metrics_h = evaluate_policy(env, policy_htrans, episodes=30, use_htrans=True, models=models)
    print(metrics_h)

    print("\n=== Summary ===")
    print("No-ESS :", metrics_no)
    print("Rule   :", metrics_rule)
    print("H-Trans:", metrics_h)


if __name__ == "__main__":
    main()
