# Hydropt
Hydropt is a dynamic prgrogramming tool for the optimization of hydro power plants. It includes classes to model power plants, their basins and turbines as well as to define scenarios with constraints.

Different scenarios can easily be compared with each other to compute and evalutate opportunity costs. A typical use case would be to price constraints due to ancillary services commitments or machine outages.

## Installation

Run the following to install:

```bash
pip install hydropt
```

## Usage

The following example shows how *hydropt* can be used for the optimization of
a simple hydro power plant with only one basin and a single turbine. 

```Python
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
```
