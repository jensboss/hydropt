import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from hydropt import Basin, Outflow, Turbine, PowerPlant, \
    Standing, MinPower, MaxPower, Scenario, Underlyings
from hydropt import get_spot_data
from hydropt.constraints import TurbineConstraint


basins = [Basin(name='basin_1', 
                volume=75e6, 
                num_states=101,
                levels=(1700, 1792),
                start_volume=60e6),
          ]

outflow = Outflow(outflow_level=1090)

turbines = [Turbine('turbine_1', 
                    max_power = 45e6,
                    base_load =  1e6,
                    efficiency=0.8, 
                    upper_basin=basins[0], 
                    lower_basin=outflow,
                    actions=[Standing(), 
                             MinPower(),
                             MaxPower()]),
            Turbine('turbine_2', 
                    max_power = 45e6,
                    base_load =  1e6,
                    efficiency=0.8,
                    upper_basin=basins[0], 
                    lower_basin=outflow,
                    actions=[Standing(), 
                             MinPower(),
                             MaxPower()])
            ]


power_plant = PowerPlant(basins, turbines)    

constraints = [TurbineConstraint(turbines[0], '2019-02-24T00', '2019-02-27T00',
                                     name='test_0', power_max=0),
               ]

market_data = load_spot_data()

n_steps = len(market_data)

time = market_data.index[0:n_steps].to_numpy().astype('datetime64[h]')
spot = market_data.iloc[0:n_steps,2].to_numpy()

inflow_rate = 5*np.ones((n_steps,1))

underlyings = Underlyings(time, spot, inflow_rate)
scenario = Scenario(power_plant, underlyings, name='base')

scenario_sdl = Scenario(power_plant, underlyings, constraints, name='SDL')

scenario.run()
scenario_sdl.run()

# print results
print(scenario.valuation(),
      scenario_sdl.valuation(),
      scenario.valuation()-scenario_sdl.valuation())

# plot results
plt.figure(2)
plt.clf()
plt.plot(time,spot, marker='.', label='hpfc')
plt.plot(time,10*inflow_rate, marker='.', label='inflow')
plt.plot(time,scenario.turbine_actions_/1e6, marker='.', label='action')
plt.plot(time,scenario.volume_[1:]/1e6, marker='.', label='vol')
plt.legend()
plt.show()

plt.figure(3)
plt.clf()
plt.plot(time,spot, marker='.', label='hpfc')
plt.plot(time,10*inflow_rate, marker='.', label='inflow')
plt.plot(time,scenario_sdl.turbine_actions_/1e6, marker='.', label='action')
plt.plot(time,scenario_sdl.volume_[1:]/1e6, marker='.', label='vol')
plt.legend()
plt.show()

plt.figure(4)
plt.clf()
plt.plot(time,spot, marker='.', label='hpfc')
plt.plot(time,
         np.sum(scenario_sdl.turbine_actions_/1e6, 1)-np.sum(scenario.turbine_actions_/1e6, 1), 
         marker='.', label='action delta')
plt.legend()
plt.show()




