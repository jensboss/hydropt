import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from hydropt.model import Basin, Outflow, Turbine, PowerPlant
from hydropt.action import Standing, MinPower, MaxPower
from hydropt.scenarios import Scenario, Underlyings
from hydropt.constraints import TurbineConstraint

basins = [Basin(name='basin_1', 
                volume=81*3600*4, 
                num_states=81, 
                init_volume=10*3600, 
                levels=(2000, 2120)),
          Basin(name='basin_2', 
                volume=31*3600*4, 
                num_states=41, 
                init_volume=10*3600, 
                levels=(1200, 1250))
          ]

outflow = Outflow(outflow_level=600)

turbines = [Turbine('turbine_1', 
                    max_power = 33e6,
                    base_load = 10e6,
                    efficiency=0.8, 
                    upper_basin=basins[0], 
                    lower_basin=basins[1]),
            Turbine('turbine_2', 
                    max_power = 15e6,
                    base_load =  7e6,
                    efficiency=0.8,
                    upper_basin=basins[1], 
                    lower_basin=outflow)
            ]

actions = [Standing(turbines[0]), 
           MinPower(turbines[0]),
           MaxPower(turbines[0]),
           Standing(turbines[1]), 
           MinPower(turbines[1]),
           MaxPower(turbines[1])
           ]

power_plant = PowerPlant(basins, turbines, actions)    

constraints = [TurbineConstraint(turbines[0], '2019-02-24T00', '2019-02-27T00',
                                     name='test_0', power_max=0),
               ]

market_data = pd.read_csv('../src/hydropt/data/spot_prices_2019.csv', 
                          sep=';', 
                          index_col=0,
                          parse_dates=True)

n_steps = 24*7*20

time = market_data.index[0:n_steps].to_numpy().astype('datetime64[h]')
spot = market_data.iloc[0:n_steps,2].to_numpy()

inflow_rate = 0.8*np.ones((n_steps,2))

underlyings = Underlyings(time, spot, inflow_rate)
scenario = Scenario(power_plant, underlyings, name='base')

scenario_sdl = Scenario(power_plant, underlyings, constraints, name='SDL')

scenario.run()
scenario_sdl.run()

#%%
print(scenario.valuation(),
      scenario_sdl.valuation(),
      scenario.valuation()-scenario_sdl.valuation())

#%%
plt.figure(2)
plt.clf()
plt.plot(time,spot, marker='.', label='hpfc')
plt.plot(time,10*inflow_rate, marker='.', label='inflow')
plt.plot(time,scenario.turbine_actions_/1e6, marker='.', label='action')
plt.plot(time,scenario.volume_[1:]/3600, marker='.', label='vol')
plt.legend()
plt.show()

plt.figure(3)
plt.clf()
plt.plot(time,spot, marker='.', label='hpfc')
plt.plot(time,10*inflow_rate, marker='.', label='inflow')
plt.plot(time,scenario_sdl.turbine_actions_/1e6, marker='.', label='action')
plt.plot(time,scenario_sdl.volume_[1:]/3600, marker='.', label='vol')
plt.legend()
plt.show()




