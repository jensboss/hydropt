"""Minimal example
"""

import numpy as np
import hydropt as ho

# Define basin
basin = ho.Basin(
    'basin_1', 
    volume=75e6, #m3 
    num_states=101, 
    levels=(1700, 1792), #m ü. M
    start_volume=60e6, #m3
)

# Define outflow
outflow = ho.Outflow(outflow_level=1090)

# Connect basin and outlow via a turbine
turbine = ho.Turbine(
    'turbine_1', 
    max_power = 45e6, #Watt
    base_load =  1e6, #Watt
    efficiency=0.8, 
    upper_basin=basin, 
    lower_basin=outflow,
    actions=[
        ho.Standing(), 
        ho.MinPower(),
        ho.MaxPower(),
    ],
)

# Define power plant model
power_plant = ho.PowerPlant([basin,], [turbine,]) 

# Load spot prices
spot_2019 = ho.load_spot_data()

# Set optimization time frame and assign price curve
time = spot_2019.index.to_numpy().astype('datetime64[h]')
spot = spot_2019['Switzerland[EUR/MWh]'].to_numpy()

# Compute inflow rate
inflow_rate = 5*np.ones((len(spot),1))

# Define a scenario
underlyings = ho.Underlyings(time, spot, inflow_rate)
scenario = ho.Scenario(power_plant, underlyings, name='base case')

# Run optimization
scenario.run()

# ... wait about 30 seconds ...

# Plot results
scenario.results_.plot()