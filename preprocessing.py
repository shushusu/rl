import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
FILE_GEN = os.path.join(BASE_PATH, "Plant_2_Generation_Data.csv")
FILE_WEATHER = os.path.join(BASE_PATH, "Plant_2_Weather_Sensor_Data.csv")

def run_preprocessing():

    p2_gen = pd.read_csv(FILE_GEN)
    p2_weather = pd.read_csv(FILE_WEATHER)

    p2_gen['DATE_TIME'] = pd.to_datetime(p2_gen['DATE_TIME'])
    p2_weather['DATE_TIME'] = pd.to_datetime(p2_weather['DATE_TIME'])

    gen_pivot = p2_gen.pivot_table(index='DATE_TIME', columns='SOURCE_KEY', values='AC_POWER', aggfunc='mean')
    
    gen_hourly = gen_pivot.resample('1h').mean().fillna(0)
    
    weather_hourly = p2_weather.set_index('DATE_TIME').resample('1h').mean(numeric_only=True)
    weather_hourly = weather_hourly.interpolate(method='linear')

    common_index = gen_hourly.index.intersection(weather_hourly.index)
    gen_hourly = gen_hourly.loc[common_index]
    weather_hourly = weather_hourly.loc[common_index]

    num_agents = gen_hourly.shape[1]
    num_steps = len(gen_hourly)

    agent_capacities = gen_hourly.max(axis=0)
    max_cap = agent_capacities.max()

    hours = gen_hourly.index.hour.values 

    base_load_profile = 500 + 2000 * np.exp(-((hours - 19)**2) / (2 * 3**2))
    
    load_matrix = np.zeros((num_steps, num_agents))
    
    np.random.seed(42)
    for i in range(num_agents):
        scale = agent_capacities.iloc[i] / max_cap
        noise = np.random.uniform(0.8, 1.2, size=num_steps)
        load_matrix[:, i] = base_load_profile * scale * noise

    price = np.where((hours >= 22) | (hours < 8), 100.0, 
            np.where((hours >= 8) & (hours < 18), 150.0, 300.0))
    
    fut_price = np.roll(price, -1); fut_price[-1] = price[-1]
    
    irrad = weather_hourly['IRRADIATION'].values
    fut_irrad = np.roll(irrad, -1); fut_irrad[-1] = irrad[-1]
    
    sin_t = np.sin(2 * np.pi * hours / 24)
    cos_t = np.cos(2 * np.pi * hours / 24)

    MAX_VALS = {
        'Gen': gen_hourly.max().max(),
        'Load': load_matrix.max(),
        'Price': 300.0,
        'Irrad': weather_hourly['IRRADIATION'].max()
    }
    
    norm_gen = gen_hourly.values / MAX_VALS['Gen']
    norm_load = load_matrix / MAX_VALS['Load']
    norm_price = np.tile(price.reshape(-1, 1), num_agents) / MAX_VALS['Price']
    norm_fut_price = np.tile(fut_price.reshape(-1, 1), num_agents) / MAX_VALS['Price']
    norm_fut_irrad = np.tile(fut_irrad.reshape(-1, 1), num_agents) / MAX_VALS['Irrad']
    
    norm_sin = np.tile(sin_t.reshape(-1, 1), num_agents) 
    norm_cos = np.tile(cos_t.reshape(-1, 1), num_agents)

    ma_data = np.stack([
        norm_gen, norm_load, norm_price, 
        norm_fut_price, norm_fut_irrad, 
        norm_sin, norm_cos
    ], axis=2)

    np.save('ma_train_data.npy', ma_data)
    print(f"shape: {ma_data.shape}")

if __name__ == "__main__":
    run_preprocessing()