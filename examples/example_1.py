import numpy as np
import matplotlib.pyplot as plt

from hydropt import Basin, Outflow, Turbine, PowerPlant, \
    Standing, MinPower, MaxPower, Scenario, Underlyings


basins = [
    Basin(
        name='Mühlensee',
        volume=81*3600,
        num_states=81,
        levels=(2000, 2120),
        start_volume=36000,
        end_volume = 10000
    ),
    Basin(
        name='Reusslifluhsee',
        volume=31*3600,
        num_states=41,
        levels=(1200, 1250),
        start_volume=36000,
        end_volume = 10000
    ),
]

outflow = Outflow(outflow_level=600, 
                  name='Vierwaldstättersee')

turbines = [
    Turbine(
        'MG 1',
        max_power=33e6,
        base_load=10e6,
        efficiency=0.8,
        upper_basin=basins[0],
        lower_basin=basins[1],
        actions=[
            Standing(),
            MinPower(),
            MaxPower()
        ]
    ),
    Turbine(
        'MG 2',
        max_power=15e6,
        base_load=7e6,
        efficiency=0.8,
        upper_basin=basins[1],
        lower_basin=outflow,
        actions=[
            Standing(),
            MinPower(),
            MaxPower()
        ]
    ),
]


power_plant = PowerPlant(basins, turbines, name='KW Pilatus')


def date_range(start_time, end_time, sampling_time=None):
    if sampling_time is None:
        start_time = np.datetime64(start_time)
        end_time = np.datetime64(end_time)
    else:
        start_time = np.datetime64(start_time, sampling_time)
        end_time = np.datetime64(end_time, sampling_time)

    return np.arange(start_time, end_time)


power_plant.summary()

time = date_range(start_time='2020-04-01T00',
                  end_time='2020-04-10')

n_steps = len(time)
hpfc = 10*(np.sin(2*np.pi*2*np.arange(n_steps)/n_steps) + 1)
inflow_rate = 0.8*np.ones((n_steps, 2))

underlyings = Underlyings(time, hpfc, inflow_rate)
scenario = Scenario(power_plant, underlyings, name='base')

scenario.run()

plt.figure(1)
plt.clf()
plt.plot(hpfc, marker='.', label='hpfc')
plt.plot(10*inflow_rate, marker='.', label='inflow')
plt.plot(scenario.turbine_actions_/1e6, marker='.', label='action')
plt.plot(np.arange(n_steps+1)-1, scenario.volume_/3600, marker='.', label='vol')
plt.legend()
plt.show()

print('EURO', np.sum(np.sum(scenario.turbine_actions_, axis=1)*hpfc)/1e6)
