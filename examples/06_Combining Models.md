# Combining Prognostic Models

This section demonstrates how prognostic models can be combined. There are two times in which this is useful: 

1. When combining multiple models of different inter-related systems into one system-of-system model (i.e., [Composite Models](https://nasa.github.io/progpy/api_ref/prog_models/CompositeModel.html)), or
2. Combining multiple models of the same system to be simulated together and aggregated (i.e., [Ensemble Models](https://nasa.github.io/progpy/api_ref/prog_models/EnsembleModel.html) or [Mixture of Expert Models](https://nasa.github.io/progpy/api_ref/progpy/MixtureOfExperts.html)). This is generally done to improve the accuracy of prediction when you have multiple models that each represent part of the behavior or represent a distribution of different behaviors. 

These two methods for combining models are described in the following sections.

## Composite Model

A CompositeModel is a PrognosticsModel that is composed of multiple PrognosticsModels. This is a tool for modeling system-of-systems. i.e., interconnected systems, where the behavior and state of one system affects the state of another system. The composite prognostics models are connected using defined connections between the output or state of one model, and the input of another model. The resulting CompositeModel behaves as a single model.

To illustrate this, we will create a composite model of an aircraft's electric powertrain, combining the DCMotor, ESC, and PropellerLoad models. The Electronic Speed Controller (ESC) converts a commanded duty (i.e., throttle) to signals to the motor. The motor then acts on the signals from the ESC to spin the load, which enacts a torque on the motor (in this case from air resistence).

First we will import the used models, and the CompositeModel class


```python
from progpy.models import DCMotor, ESC, PropellerLoad
from progpy import CompositeModel
```

Next we will initiate objects of the individual models that will later create the composite powertrain model.


```python
m_motor = DCMotor()
m_esc = ESC()
m_load = PropellerLoad()
```

Next we have to define the connections between the systems. Let's first define the connections from the DCMotor to the propeller load. For this, we'll need to look at the DCMotor states and understand how they influence the PropellerLoad inputs.


```python
print('motor states: ', m_motor.states)
print('load inputs: ', m_load.inputs)
```

Each of the states and inputs are described in the model documentation at [DC Motor Docs](https://nasa.github.io/progpy/api_ref/prog_models/IncludedModels.html#dc-motor) and [Propeller Docs](https://nasa.github.io/progpy/api_ref/prog_models/IncludedModels.html#propellerload)

From reading the documentation we understand that the propeller's velocity is from the motor, so we can define the first connection:


```python
connections = [
    ('DCMotor.v_rot', 'PropellerLoad.v_rot')
]
```

Connections are defined as couples where the first value is the input for the second value. The connection above tells the composite model to feed the DCMotor's v_rot into the PropellerLoad's input v_rot.

Next, let's look at the connections the other direction, from the load to the motor.


```python
print('load states: ', m_load.states)
print('motor inputs: ', m_motor.inputs)
```

We know here that the load on the motor is from the propeller load, so we can add that connection. 


```python
connections.append(('PropellerLoad.t_l', 'DCMotor.t_l'))
```

Now we will repeat the exercise with the DCMotor and ESC.


```python
print('ESC states: ', m_esc.states)
print('motor inputs: ', m_motor.inputs)
connections.append(('ESC.v_a', 'DCMotor.v_a'))
connections.append(('ESC.v_b', 'DCMotor.v_b'))
connections.append(('ESC.v_c', 'DCMotor.v_c'))

print('motor states: ', m_motor.states)
print('ESC inputs: ', m_esc.inputs)
connections.append(('DCMotor.theta', 'ESC.theta'))
```

Now we are ready to combine the models. We create a composite model with the inidividual models and the defined connections.


```python
m_powertrain = CompositeModel(
        (m_esc, m_load, m_motor), 
        connections=connections)
```

The resulting model includes two inputs, ESC voltage (from the battery) and duty (i.e., commanded throttle). These are the only two inputs not connected internally from the original three models. The states are a combination of all the states of every system. Finally, the outputs are a combination of all the outputs from each of the individual systems. 


```python
print('inputs: ', m_powertrain.inputs)
print('states: ', m_powertrain.states)
print('outputs: ', m_powertrain.outputs)
```

Frequently users only want a subset of the outputs from the original model. For example, in this case you're unlikely to be measuring the individual voltages from the ESC. Outputs can be specified when creating the composite model. For example:


```python
m_powertrain = CompositeModel(
        (m_esc, m_load, m_motor), 
        connections=connections,
        outputs={'DCMotor.v_rot', 'DCMotor.theta'})
print('outputs: ', m_powertrain.outputs)
```

Now the outputs are only DCMotor angle and velocity.

The resulting model can be used in simulation, state estimation, and prediction the same way any other model would be, as demonstrated below:


```python
load = m_powertrain.InputContainer({
        'ESC.duty': 1, # 100% Throttle
        'ESC.v': 23
    })
def future_loading(t, x=None):
    return load

simulated_results = m_powertrain.simulate_to(1, future_loading, dt=2.5e-5, save_freq=2e-2)
fig = simulated_results.outputs.plot(compact=False, keys=['DCMotor.v_rot'], ylabel='Velocity')
fig = simulated_results.states.plot(keys=['DCMotor.i_b', 'DCMotor.i_c', 'DCMotor.i_a'], ylabel='ESC Currents')
```

Parameters in composed models can be updated directly using the model_name.parameter name parameter of the composite model. Like so:


```python
m_powertrain.parameters['PropellerLoad.D'] = 1
```

Here we updated the propeller diameter to 1, greatly increasing the load on the motor. You can see this in the updated simulation outputs (below). When compared to the original results above you will find that the maximum velocity is lower. This is expected given the larger propeller load.


```python
simulated_results = m_powertrain.simulate_to(1, future_loading, dt=2.5e-5, save_freq=2e-2)
fig = simulated_results.outputs.plot(compact=False, keys=['DCMotor.v_rot'], ylabel='Velocity')
fig = simulated_results.states.plot(keys=['DCMotor.i_b', 'DCMotor.i_c', 'DCMotor.i_a'], ylabel='ESC Currents')
```

Note: A function can be used to perform simple transitions between models. For example, if you wanted to multiply the torque by 1.1 to represent some gearing or additional load, that could be done by defining a function, as follows


```python
def torque_multiplier(t_l):
    return t_l * 1.1
```

The function is referred to as 'function' by the composite model. So we can add the function into the connections as follows. Note that the argument name is used for the input of the function and 'return' is used to signify the function's return value. 


```python
connections = [
    ('PropellerLoad.t_l', 'function.t_l'),
    ('function.return', 'DCMotor.t_l')
]
```

Now let's add back in the other connections and build the composite model


```python
connections.extend([
    ('ESC.v_a', 'DCMotor.v_a'),
    ('ESC.v_b', 'DCMotor.v_b'),
    ('ESC.v_c', 'DCMotor.v_c'),
    ('DCMotor.theta', 'ESC.theta'),
    ('DCMotor.v_rot', 'PropellerLoad.v_rot')
])
m_powertrain = CompositeModel(
        (m_esc, m_load, m_motor, torque_multiplier), 
        connections=connections,
        outputs={'DCMotor.v_rot', 'DCMotor.theta'})
simulated_results = m_powertrain.simulate_to(1, future_loading, dt=2.5e-5, save_freq=2e-2)
fig = simulated_results.outputs.plot(compact=False, keys=['DCMotor.v_rot'], ylabel='Velocity')
fig = simulated_results.states.plot(keys=['DCMotor.i_b', 'DCMotor.i_c', 'DCMotor.i_a'], ylabel='ESC Currents')
```

Note that you can also have functions with more than one argument. If you dont connect the arguments of the function to some model, it will show up in the inputs of the composite model.

## Ensemble Model

An ensemble model is an approach to modeling where one or more models of the same system are simulated together and then aggregated into a single prediction. This can be multiple versions of the same model with different parameters, or different models of the same system representing different parts of the system's behavior. This is generally done to improve the accuracy of prediction when you have multiple models that each represent part of the behavior or represent a distribution of different behaviors.

In ensemble models, aggregation occurs in two steps, at state transition and then output, event state, threshold met, or performance metric calculation. At each state transition, the states from each aggregate model are combined based on the defined aggregation method. When calling output, the resulting outputs from each aggregate model are similarily combined. The default method is mean, but the user can also choose to use a custom aggregator.

![Aggregation](img/aggregation.png)

To illustrate this, let's create an example where there we have four equivalent circuit models, each with different configuration parameters, below. These represent the range of possible configurations expected for our example system.


```python
from progpy.models import BatteryCircuit
m_circuit = BatteryCircuit()
m_circuit_2 = BatteryCircuit(qMax = 7860)
m_circuit_3 = BatteryCircuit(qMax = 6700, Rs = 0.055)
```

Let's create an EnsembleModel which combines each of these.


```python
from progpy import EnsembleModel
m_ensemble = EnsembleModel(
    models=(m_circuit, m_circuit_2, m_circuit_3))
```

Now let's evaluate the performance of the combined model using real battery data from NASA's prognostic data repository. See 07. Datasets for more detail on accessing data from this repository


```python
from progpy.datasets import nasa_battery
data = nasa_battery.load_data(batt_id=8)[1]
RUN_ID = 0
test_input = [{'i': i} for i in data[RUN_ID]['current']]
test_time = data[RUN_ID]['relativeTime']
```

To evaluate the model we first create a future loading function that uses the loading from the data.


```python
def future_loading(t, x=None):
    for i, mission_time in enumerate(test_time):
        if mission_time > t:
            return m_circuit.InputContainer(test_input[i])
    return m_circuit.InputContainer(test_input[-1])  # Default - last load
```


```python
t_end = test_time.iloc[-1]
from matplotlib import pyplot as plt
```

Next we will simulate the ensemble model


```python
t_end = test_time.iloc[-1]
results_ensemble = m_ensemble.simulate_to(t_end, future_loading)
```

Finally, we compare the voltage predicted by the ensemble model with the ground truth from dataset.


```python
from matplotlib import pyplot as plt
fig = plt.plot(test_time, data[RUN_ID]['voltage'], color='green', label='ground truth')
fig = plt.plot(results_ensemble.times, [z['v'] for z in results_ensemble.outputs], color='red', label='ensemble')
plt.xlabel('Time (s)')
plt.ylabel('Voltage')
plt.legend()
```

The ensemble model actually performs pretty poorly here. This is mostly because there's an outlier model (m_circuit_3). This can be resolved using a different aggregation method. By default, aggregation uses the mean. Let's update the ensemble model to use median and resimulate


```python
import numpy as np
m_ensemble['aggregation_method'] = np.median

results_ensemble_median = m_ensemble.simulate_to(t_end, future_loading)
fig = plt.plot(results_ensemble_median.times, [z['v'] for z in results_ensemble_median.outputs], color='orange', label='ensemble -median')
fig = plt.plot(test_time, data[RUN_ID]['voltage'], color='green', label='ground truth')
fig = plt.plot(results_ensemble.times, [z['v'] for z in results_ensemble.outputs], color='red', label='ensemble')
plt.xlabel('Time (s)')
plt.ylabel('Voltage')
plt.legend()
```

Much better!

The same ensemble approach can be used with a heterogeneous set of models that have different states.

Here we will repeat the exercise using the battery electrochemisty and equivalent circuit models. The two models share one state in common (tb), but otherwise are different


```python
from progpy.models import BatteryElectroChemEOD
m_electro = BatteryElectroChemEOD(qMobile=7800)

print('Electrochem states: ', m_electro.states)
print('Equivalent Circuit States', m_circuit.states)
```

Now let's create an ensemble model combining these and evaluate it.


```python
m_ensemble = EnsembleModel((m_circuit, m_electro))
results_ensemble = m_ensemble.simulate_to(t_end, future_loading)
```

To compare these results, let's also simulate the two models that comprise the ensemble model.


```python
results_circuit1 = m_circuit.simulate_to(t_end, future_loading)
results_electro = m_electro.simulate_to(t_end, future_loading)
```

The results of each of these are plotted below.


```python
plt.figure()
plt.plot(results_circuit1.times, [z['v'] for z in results_circuit1.outputs], color='blue', label='circuit')
plt.plot(results_electro.times, [z['v'] for z in results_electro.outputs], color='red', label='electro chemistry')
plt.plot(results_ensemble.times, [z['v'] for z in results_ensemble.outputs], color='yellow', label='ensemble')
plt.plot(test_time, data[RUN_ID]['voltage'], color='green', label='ground truth')
plt.legend()
```

Note that the result may not be exactly between the other two models. This is because of aggregation is done in 2 steps: at state transition and then at output calculation.

Ensemble models can be further extended to include an aggregator that selects the best model at any given time. That feature is described in the following section.

## Mixture of Experts (MoE)

Mixture of Experts (MoE) models combine multiple models of the same system, similar to Ensemble models. Unlike Ensemble Models, the aggregation is done by selecting the "best" model. That is the model that has performed the best over the past. Each model will have a 'score' that is tracked in the state, and this determines which model is best.

To demonstrate this feature we will repeat the example from the ensemble model section, this time with a mixture of experts model. For this example to work you will have had to have run the ensemble model section example.

First, let's combine the three battery circuit models into a single mixture of experts model.


```python
from progpy import MixtureOfExpertsModel
m = MixtureOfExpertsModel((m_circuit_3, m_circuit_2, m_circuit))
```

The combined model has the same outputs and events as the circuit model. 


```python
print(m.outputs)
print(m.events)
```

Its states contain all of the states of each model, kept separate. Each individual model comprising the MoE model will be simulated separately, so the model keeps track of the states propogated through each model separately. The states also include scores for each model.


```python
print(m.states)
```

The MoE model inputs include both the comprised model input, `i` (current) and outputs: `v` (voltage) and `t`(temperature). The comprised model outputs are provided to update the scores of each model when performing state transition. If they are not provided when calling next_state, then scores would not be updated.


```python
print(m.inputs)
```

Now let's evaluate the performance of the combined model using real battery data from NASA's prognostic data repository, downloaded in the previous sections. See 07. Datasets for more detail on accessing data from this repository.

To evaluate the model we first create a future loading function that uses the loading from the data.


```python
results_moe = m.simulate_to(t_end, future_loading)
fig = plt.plot(test_time, data[RUN_ID]['voltage'], color='green', label='ground truth')
fig = plt.plot(results_moe.times, [z['v'] for z in results_moe.outputs], color='red', label='ensemble')
plt.xlabel('Time (s)')
plt.ylabel('Voltage')
plt.legend()
```

Here the model performs pretty poorly. If you were to look at the state, we see that the three scores are equal. This is because we haven't provided any output information. The future load function doesn't include the output, just the input (`i`). When the three scores are equal like this, the first model is used.


```python
print('Model 1 Score: ', results_moe.states[-1]['BatteryCircuit._score'])
print('Model 2 Score: ', results_moe.states[-1]['BatteryCircuit_2._score'])
print('Model 3 Score: ', results_moe.states[-1]['BatteryCircuit_3._score'])
```

Now let's provide the output for a few steps.


```python
x0 = m.initialize()
x = m.next_state(
    x=x0, 
    u=m.InputContainer({
        'i': test_input[0]['i'],
        'v': data[RUN_ID]['voltage'][0],
        't': data[RUN_ID]['temperature'][0]}),
    dt=test_time[1]-test_time[0])
x = m.next_state(
    x=x, 
    u=m.InputContainer({
        'i': test_input[1]['i'],
        'v': data[RUN_ID]['voltage'][1],
        't': data[RUN_ID]['temperature'][1]}),
    dt=test_time[1]-test_time[0])
```

Let's take a look at the model scores again


```python
print('Model 1 Score: ', x['BatteryCircuit._score'])
print('Model 2 Score: ', x['BatteryCircuit_2._score'])
print('Model 3 Score: ', x['BatteryCircuit_3._score'])
```

Here we see after a few steps the algorithm has determined that model 3 is the better fitting of the models. Now if we were to repeat the simulation, it would use the best model, resulting in a better fit. 


```python
results_moe = m.simulate_to(t_end, future_loading, t0=test_time[1]-test_time[0], x=x)
fig = plt.plot(test_time[2:], data[RUN_ID]['voltage'][2:], color='green', label='ground truth')
fig = plt.plot(results_moe.times[2:], [z['v'] for z in results_moe.outputs][2:], color='red', label='moe')
plt.xlabel('Time (s)')
plt.ylabel('Voltage')
plt.legend()
```

The fit here is much better. The MoE model learned which of the three models best fit the observed behavior.

In a prognostic application, the scores will be updated each time you use a state estimator (so long as you provide the output as part of the input). Then when performing a prediction the scores aren't updated, since outputs are not known.

An example of when this would be useful is for cases where there are three common degradation paths or "modes" rather than a single model with uncertainty to represent every mode, the three modes can be represented by three different models. Once enough of the degradation path has been observed the observed mode will be the one reported.

If the model fit is expected to be stable (that is, the best model is not expected to change anymore). The best model can be extracted and used directly, like demonstrated below.


```python
name, m_best = m.best_model(x)
print(name, " was the best fit")
```

## Conclusions

In this section we demonstrated a few methods for treating multiple models as a single model. This is of interest when there are multiple models of different systems which are interdependent (CompositeModel), multiple models of the same system that portray different parts of the behavior or different candidate representations (EnsembleModel), or multiple models of the same system that represent possible degradation modes (MixtureOfExpertModel).
