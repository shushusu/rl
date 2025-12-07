import numpy as np
import os

class HierarchicalESSEnv:
    def __init__(
        self,
        data_path='ma_train_data.npy',
        is_train=True,
        cost_scale=0.1,
        alpha_peak=0.5,
        beta_peak=0.25,
    ):
        
        self.data = np.load(data_path).astype(np.float32)  
        self.n_steps, self.n_agents, self.n_feats = self.data.shape
        self.is_train = is_train
        
        self.BATTERY_CAP = 2.0  
        self.MAX_POWER = 0.5    
        self.EFFICIENCY = 0.9   
        self.MAX_EPISODE_STEPS = 24 
        
        self.PEAK_THRESHOLD = 12.0

        self.COST_SCALE = cost_scale  
        self.ALPHA_PEAK = alpha_peak   
        self.BETA_PEAK  = beta_peak    

        self.mgr_obs_dim = 7 
        self.wrk_obs_dim = 9
        
        self.mgr_act_dim = 1 
        self.wrk_act_dim = 1 

        self.soc = np.zeros(self.n_agents, dtype=np.float32)
        self.timestep = 0
        self.start_step = 0

    def reset(self):
        if self.is_train:
            self.start_step = np.random.randint(0, self.n_steps - self.MAX_EPISODE_STEPS - 1)
            self.soc = np.random.uniform(0.2, 0.8, size=self.n_agents).astype(np.float32)
        else:
            self.start_step = 0
            self.soc = np.full(self.n_agents, 0.5, dtype=np.float32)
            
        self.timestep = self.start_step
        return self._get_obs(mgr_signal=1.0)

    def _get_obs(self, mgr_signal):
        step_data = self.data[self.timestep]  
        mgr_obs = np.mean(step_data, axis=0)  
        soc_col = self.soc.reshape(-1, 1)
        signal_col = np.full((self.n_agents, 1), mgr_signal)
        wrk_obs = np.hstack([soc_col, step_data, signal_col])  
        return mgr_obs.astype(np.float32), wrk_obs.astype(np.float32)

    def step(self, mgr_action, wrk_actions):
        limit_signal = np.clip(mgr_action, 0.0, 1.0)
        raw_wrk_actions = np.clip(wrk_actions.flatten(), -1.0, 1.0)
        
        constrained_actions = np.where(
            raw_wrk_actions < 0, 
            raw_wrk_actions * limit_signal, 
            raw_wrk_actions 
        )
        
        actual_power = np.zeros_like(constrained_actions)
        
        charge_idx = constrained_actions > 0
        max_charge = (1.0 - self.soc) * self.BATTERY_CAP / self.EFFICIENCY
        real_charge = np.minimum(constrained_actions * self.MAX_POWER, max_charge)
        self.soc[charge_idx] += (real_charge[charge_idx] * self.EFFICIENCY) / self.BATTERY_CAP
        actual_power[charge_idx] = real_charge[charge_idx]

        discharge_idx = constrained_actions < 0
        max_discharge = self.soc * self.BATTERY_CAP * self.EFFICIENCY
        real_discharge = np.minimum(-constrained_actions * self.MAX_POWER, max_discharge)
        self.soc[discharge_idx] -= (real_discharge[discharge_idx] / self.EFFICIENCY) / self.BATTERY_CAP
        actual_power[discharge_idx] = -real_discharge[discharge_idx]
        
        self.soc = np.clip(self.soc, 0.0, 1.0)

        gen = self.data[self.timestep, :, 0]
        load = self.data[self.timestep, :, 1]
        price = self.data[self.timestep, :, 2]
        
        net_load = load - gen
        grid_power = net_load + actual_power          
        total_grid_load = np.sum(grid_power)

        costs = np.maximum(0, grid_power) * price     
        total_cost = float(np.sum(costs))             

        scaled_costs_each = costs * self.COST_SCALE
        scaled_total_cost = float(np.sum(scaled_costs_each))

        peak_penalty = 0.0
        if abs(total_grid_load) > self.PEAK_THRESHOLD:
            peak_penalty = abs(abs(total_grid_load) - self.PEAK_THRESHOLD)

        mgr_reward = -scaled_total_cost - self.ALPHA_PEAK * peak_penalty
        wrk_rewards = -scaled_costs_each - self.BETA_PEAK * peak_penalty

        self.timestep += 1
        done = False
        if self.timestep >= self.start_step + self.MAX_EPISODE_STEPS:
            done = True
            
        next_mgr_obs, next_wrk_obs = self._get_obs(mgr_signal=limit_signal)
        
        info = {
            'total_load': float(total_grid_load),
            'peak_penalty': float(peak_penalty),
            'limit_signal': float(limit_signal),
            'total_cost': total_cost,  
        }
        
        return (next_mgr_obs, next_wrk_obs), (mgr_reward, wrk_rewards), done, info
