# Welcome to the Prognostics Python Package (ProgPy) Tutorial

The goal of this notebook is to instruct users on how to use and extend the NASA PCoE Python Prognostics Python Package (progpy). 

First some background. The Prognostics Python Package is a python package for the modeling, simulation, state estimation, and prediction of the evolution of state for components, systems, and systems of systems, with a focus on simulating specific events. When used for prognostics, these events are typically system failures, such as a winding failure on a motor or full discharge of a battery. 

A few definitions:
* __Event__: Something that can be predicted (e.g., system failure, warning threshold). An event has either occurred or not. 
* __Event State__: Progress towards event occurring. Defined as a number where an event state of 0 indicates the event has occurred and 1 indicates no progress towards the event (i.e., fully healthy operation for a failure event). For gradually occurring events (e.g., discharge) the number will progress from 1 to 0 as the event nears. In prognostics, event state is frequently called "State of Health" or "SOH"
* __Inputs__: Control applied to the system being modeled (e.g., current drawn from a battery)
* __Outputs__: Measured sensor values from a system (e.g., voltage and temperature of a battery), outputs can be estimated from system state
* __States__: Internal parameters (typically hidden states) used to represent the state of the system- can be the same as inputs/outputs but do not have to be.
* __Performance Metrics__: Performance characteristics of a system that are a function of system state, but are not directly measured. For example, a performance metric for a electric motor might be the maximum achievable torque 
* __State Estimation__: The process of estimating the (possibly hidden) state of a system given sensor information on observable states
* __Prediction__: The process of predicting the evolution of a system state with time and the occurrence of events. 

The `progpy` package has the following structure
* `progpy.data_models` - package containing algorithms for data-driven models, and parent class `progpy.data_models.DataModel`
* `progpy.datasets` - package containing tools for downloading a few relevant datasets
* `progpy.loading` - package containing tools for representing different loading profiles
* `progpy.models.*` - implemented models (e.g., pump, valve, battery)
* `progpy.state_estimators.*` - State Estimation algorithms
* `progpy.predictors.*` - Prediction algorithms
* `progpy.uncertain_data.*` - Classes for representing data with uncertainty
* `progpy.utils.*` - various utility functions and classes
* `progpy.CompositeModel` - model of a system-of-systems, combining multiple interrelated models into a single model
* `progpy.EnsembleModel` - model combining multiple models of the same system into a single model
* `progpy.MixtureOfExpertsModel` - model combining multiple models of the same system into a single model where the "best" model is used at every step
* `progpy.LinearModel` - parent class for simple linear models
* `progpy.PrognosticsModel` - parent class for all prognostics models - defines interfaces that a model must implement, and tools for simulating said model

In addition to the `proypy` package, the GitHub repository includes many examples illustrating how to use the package (see `examples/`), a template for implementing a new model (`prog_model_template`), a template for implementing a new state estimator (`state_estimator_template`), a template for implementing a new predictor (`predictor_template`), and this tutorial (`tutorial.ipynb`). Documentation  for ProgPy can be found at <https://nasa.github.io/progpy>,

Before you start, make sure to install progpy using the following command:

    pip install progpy

Now let's get started with some examples

## Using the included models

This first example is for using the included prognostics models. 

The `progpy.models` package includes implemented models, including ones for pumps, valves, batteries, and more. See <https://nasa.github.io/progpy/api_ref/progpy/IncludedModels.html> for a full description of the included models.

First thing to do is to import the model you would like to use:


```python
from progpy.models import BatteryCircuit
```

This imports the BatteryCircuit model distributed with the `progpy` package. See <https://nasa.github.io/progpy/api_ref/progpy/IncludedModels.html> for details on this model.

Next, let's create a new battery using the default settings:


```python
batt = BatteryCircuit()
```

This creates a battery circuit model. You can also pass configuration parameters into the constructor as keyword arguments to configure the system, for example
### <center>`BatteryCircuit(qMax = 7856)`</center>
Alternatively, you can set the configuration of the system afterwards, like so:


```python
batt.parameters['qMax'] = 7856 
batt.parameters['process_noise'] = 0  # Note: by default there is some process noise- this turns it off. Noise will be explained later in the tutorial
```

These parameters describe the specific system (in this case battery) the model is simulating. See <https://nasa.github.io/progpy/api_ref/progpy/IncludedModels.html> for a full list of configurable model parameters. Let's use the model properties to check out some details of this model, first the model configuration:


```python
from pprint import pprint
print('Model configuration:')
pprint(batt.parameters)
```

You can save or load your model configuration using pickle


```python
import pickle
pickle.dump(batt.parameters, open('battery123.cfg', 'wb'))
```

Then you can set your model configuration like below. This is useful for recording and restoring model configurations. Some users store model configuration as picked files to be restored from later.


```python
batt.parameters = pickle.load(open('battery123.cfg', 'rb'))
```

Additionally, both pickle and JSON can be used to serialize an entire model for future use. All PrognosticsModels include functions to serialize with JSON as follows:


```python
save_json = batt.to_json() # Save model 
serial_batt = BatteryCircuit.from_json(save_json) # Model can be called directly with serialized result
```

Information is passed to and from the model using containers that function like dictionaries. The keys of the containers are specific to the model.

Let's look at the inputs (loading) and outputs (measurements) for a BatteryCircuit model. These are the keys expected for inputs and outputs, respectively.


```python
print('inputs: ', batt.inputs)
print('outputs: ', batt.outputs)
```

Now let's look at what events we're predicting. This model only predicts one event, called EOD (End of Discharge), but that's not true for every model. See <https://nasa.github.io/progpy/models.html>


```python
print('event(s): ', batt.events)
```

Finally, let's take a look at the internal states that the model uses to represent the system:


```python
print('states: ', batt.states)
```

## Simulating to a specific time

Now let's simulate. The following section uses the model created in the "using the included models" section.

First, we define future loading. This is a function that describes how the system will be loaded as a function of time. Here we're defining a basic piecewise function.


```python
from progpy.loading import Piecewise
# Variable (piece-wise) future loading scheme 
future_loading = Piecewise(batt.InputContainer, [600, 900, 1800, 3000, float('inf')], {'i': [2, 1, 4, 2, 3]})
```

Note that future loading can be modeled using various classes in progpy.loading or as any function f(t: float, x: StateContainer) -> InputContainer.

At last it's time to simulate. First we're going to simulate forward 200 seconds. To do this we use the function simulate_to() to simulate to the specified time and print the results.


```python
time_to_simulate_to = 200
sim_config = {
    'save_freq': 20, 
    'print': True  # Print states - Note: is much faster without
}
(times, inputs, states, outputs, event_states) = batt.simulate_to(time_to_simulate_to, future_loading, **sim_config)
```

We can also plot the results. Here we see the temperature of the battery increase and the voltage decrease with use. This is expected. Voltage will decrease as the state of charge decreases, and temperature increases as current is drawn through the battery, until it reaches some equilibrium. Everything is very linear because the load is kept constant within the simulation window. 


```python
inputs.plot(ylabel='Current drawn (amps)')
event_states.plot(ylabel= 'SOC')
outputs.plot(ylabel= {'v': "voltage (V)", 't': 'temperature (°C)'}, compact= False)
```

Also, note the lack of smoothness in the voltage curve. This is limited by the save_freq from sim_cfg. There is a point on the graph every 20 seconds, because that is the frequency at which we save the results. Decreasing the save frequency will result in a cleaner curve.

The results can be further analyzed with available metrics. For example, monotonicity can be calculated for simulate_to()'s returned objects. The End of Discharge (EOD) event state (i.e., State of Charge) should be monotonic (i.e., decreasing monotonically). Note: the EOD event is specific to the battery model. Each model will simulate different events.

The monotonicity metric indicates the degree of monotonicity where 1 is completely monotonic and 0 is perfectly non-monotonic (decreasing as much as increasing)


```python
print('monotonicity: ', event_states.monotonicity())
```

Lastly, results can be stored in a container variable and be individually accessed via namedtuple syntax.


```python
batt_simulation = batt.simulate_to(time_to_simulate_to, future_loading, save_freq = 20)
print('times: ', batt_simulation.times) 
print('\ninputs: ', batt_simulation.inputs)
print('\nstates: ', batt_simulation.states)
print('\noutputs: ', batt_simulation.outputs) 
print('\nevent states: ', batt_simulation.event_states)
```

## Simulating to threshold

Instead of specifying a specific amount of time, we can also simulate until a threshold has been met using the simulate_to_threshold() method. Results can be similarly plotted and accessed via namedtuple syntax.


```python
options = { #configuration for this sim
    'save_freq': 100,  # Frequency at which results are saved (s)
    'horizon': 5000  # Maximum time to simulate (s) - This is a cutoff. The simulation will end at this time, or when a threshold has been met, whichever is first
    }
(times, inputs, states, outputs, event_states) = batt.simulate_to_threshold(future_loading, **options)
inputs.plot(ylabel='Current drawn (amps)')
event_states.plot(ylabel='SOC')
outputs.plot(ylabel= {'v': "voltage (V)", 't': 'temperature (°C)'}, compact= False)
```

One thing to note here is that unlike the earlier plots, these plots are not smooth curves, this is because the load is piecewise, not constant. You see jumps in the plots at the times when the load changes. Also, the simulation is long enough for the temperature to reach an equilibrium. 

Default is to simulate until any threshold is met, but we can also specify which event we are simulating to (any key from model.events) for multiple event models. See `examples.sim_battery_eol` for an example of this.

## Noise

There are three types of noise that can be added to a model in simulation, described below:
* __Process Noise__: Noise representing uncertainty in the model transition; e.g., model or model configuration uncertainty, uncertainty from simplifying assumptions. Applied during state transition
* __Measurement Noise__: Noise representing uncertainty in the measurement process; e.g., sensor sensitivity, sensor misalignments, environmental effects. Applied during estimation of outputs from states.
* __Future Loading Noise__: Noise representing uncertainty in the future loading estimates; e.g., uncertainty from incomplete knowledge of future loading. Responsibility of user to apply in supplied future loading method

The amount of process or measurement noise is considered a property of the model and can be set using the m.parameters['process_noise'] and m.parameters['measurement_noise'], respectively.

In this example we will use the ThrownObject model and turn off process noise. ThrownObject is a simple model of an object thrown directly into the air in a vacuum. Thrown object simulates two events: 'falling' (when the object starts falling) and 'impact' (when the object hits the ground). More details can be found later in the tutorial.


```python
from progpy.models import ThrownObject

# Create an instance of the thrown object model with no process noise
m = ThrownObject(process_noise=False)

# Define future loading
def future_load(t=None, x=None):  
    # The thrown object model has no inputs- you cannot load the system (i.e., effect it once it's in the air)
    # So we return an empty input container
    return m.InputContainer({})

# Define configuration for simulation
config = {
    'events': 'impact', # Simulate until the thrown object has impacted the ground
    'dt': 0.005, # Time step (s)
    'save_freq': 0.5, # Frequency at which results are saved (s)
}

# Simulate to a threshold
(times, _, states, outputs, _) = m.simulate_to_threshold(future_load, **config)

# Print results
print('states:')
for (t,x) in zip(times, states):
    print('\t{:.2f}s: {}'.format(t, x))

print('\nimpact time: {:.2f}s'.format(times[-1]))
# The simulation stopped at impact, so the last element of times is the impact time

# Plot results
states.plot()
```

See above the velocity decreases linearly and the position follows a clean parabola, as we would expect.

Now with this second example we apply normal (i.e., gaussian) process noise with a standard deviation of 15 to every state. Compare the plots generated with those above- you should be able to see the effects of the noise


```python
m = ThrownObject(process_noise = 15)

# Simulate to a threshold
(times, _, states, outputs, _) = m.simulate_to_threshold(future_load, **config)

# Print Results
print('states:')
for (t,x) in zip(times, states):
    print('\t{:.2f}s: {}'.format(t, x))

print('\nimpact time: {:.2f}s'.format(times[-1]))

# Plot results
states.plot()
```

You can also specify different amounts of noise on different states, for example here we are applying no noise to velocity but a large amount of noise to the position. Compare the plot with that above. Here you should see a smooth curve for the velocity, but a noisy curve for position.


```python
m = ThrownObject(process_noise = {'x': 50, 'v': 0})

# Simulate to a threshold
(times, _, states, outputs, _) = m.simulate_to_threshold(future_load, **config)

# Print Results
print('states:')
for (t,x) in zip(times, states):
    print('\t{:.2f}s: {}'.format(t, x))

print('\nimpact time: {:.2f}s'.format(times[-1]))

# Plot results
states.plot()
```

You can also define the shape of the noise to be uniform or triangular instead of normal. Finally, you can define your own function to apply noise for anything more complex. 

For more information see `examples.noise`

## Simulating - advanced

There are a number of advanced features that can be used in simulation. A few of these will be described below. Detail can also be found in the documentation here: https://nasa.github.io/progpy/prog_models_guide.html#simulation

### Saving results at specific points

Earlier, we demonstrated how save_freq specifies the frequency at which the results are saved in simulation. There are occasionally circumstances when you need the results at a specific time (e.g., check points, waypoints, etc.). This is accomplished with the save_pts argument in simulation.

For example, in the following simple case, we're reusing the example from earlier, but we're saving the results at 1 second, 5 seconds, 6 seconds, and 7 seconds. 


```python
# Reset noise
m.parameters['process_noise'] = False
m.parameters['measurement_noise'] = False

# Simulate to a threshold
(times, _, states, outputs, _) = m.simulate_to_threshold(future_load, dt=config['dt'], events='impact', save_pts=[1, 5, 6, 7])

# Print Results
print('states:')
for (t,x) in zip(times, states):
    print('\t{:.2f}s: {}'.format(t, x))

# Plot results
states.plot()
```

Note that now we only have the data at the points specified (plus initial and final). As a result, the graph looks lopsided.

save_pts can also be used to adjust the save frequency as degradation progresses, for example by having more concentrated save pts around the end of degradation, like the example below.


```python
# Simulate to a threshold
(times, _, states, outputs, _) = m.simulate_to_threshold(future_load, dt=config['dt'], events='impact', save_pts=[1, 2, 3, 4, 5, 5.5, 6, 6.25, 6.5, 6.75, 7, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8])

# Print Results
print('states:')
for (t,x) in zip(times, states):
    print('\t{:.2f}s: {}'.format(t, x))

# Plot results
states.plot()
```

Now the curve is much better defined as the simulation progresses. This is frequently of interest since users are mostly interested in behavior close to the event occurrence.

Keep in mind that using a fixed dt, like above, will only save the data at the closest dt increment at or after the save point. So if save_freq is less than dt, save points will be missed. Or, if save_pts are not a multiple of dt, the resulting point at which data is saved will not be exactly at the save point. Auto step sizes (see next section) can help with this.

### Step Sizes

At the end of the save_pts example above, we noted that save_pts are often not exactly met when they are not a multiple of dt. This can be seen in the example below.


```python
# Simulate to a threshold
(times, _, states, outputs, _) = m.simulate_to_threshold(future_load, dt=0.2, events='impact', save_pts=[1, 2.5, 3, 4.5, 6, 7.5])

# Print Results
print('states:')
for (t,x) in zip(times, states):
    print('\t{:.2f}s: {}'.format(t, x))

# Plot results
states.plot()
```

Note above that the state was captured at 2.6, 4.6, and 7.6 instead of the requested 2.5, 4.5, and 7.5. 

This can be fixed by auto step sizes. Let's repeat this example with automatic step sizes.


```python
# Simulate to a threshold
(times, _, states, outputs, _) = m.simulate_to_threshold(future_load, dt='auto', events='impact', save_pts=[1, 2.5, 3, 4.5, 6, 7.5])

# Print Results
print('states:')
for (t,x) in zip(times, states):
    print('\t{:.2f}s: {}'.format(t, x))

# Plot results
states.plot()
```

Now the save_pts are captured exactly, but note that the resulting behavior is very different than the results of previous simulations. This is because we used an unbounded automatic step size. The simulation took larger steps than we should, resulting in error accumulation.

With bounded automatic step sizes we also provide the largest step it can take. This prevents the integrator from taking too large steps.


```python
# Simulate to a threshold
(times, _, states, outputs, _) = m.simulate_to_threshold(future_load, dt=('auto', 0.2), events='impact', save_pts=[1, 2.5, 3, 4.5, 6, 7.5])

# Print Results
print('states:')
for (t,x) in zip(times, states):
    print('\t{:.2f}s: {}'.format(t, x))

# Plot results
states.plot()
```

Now the simulation still captures every save point exactly, but it never takes a step size larger than 0.2 seconds, resulting in a better simulation.

You can also define more complex dynamic step sizes by providing a function (t, x)->dt. For example: `dt=lambda t, x: max(0.75 - t/100, 0.25)`

### Integration methods

Simulation is essentially the process of integrating the model forward with time. By default, simple Euler integration is used to propagate the model forward. Advanced users can change the numerical integration method to affect the simulation quality and runtime. This is done using the `integration_method` argument in `simulate_to()`, `simulate_to_threshold()`, or the model parameters. For example:

`m.parameters['integration_method'] = 'rk4'`

Note: integration method can only be changed for continuous models. In this case our thrown object model is discrete, so we cannot demonstrate it on this model (see below).


```python
m.is_continuous
```

Integration methods can also be set to scipy.integrate functions, for example:

`m.parameters['integration_method'] = scipy.integrate.Radau`

Any keyword arguments are then saved into `m.parameters['integrator_config']` as a dictionary

## Building a new model

To build a model, create a seperate class to define the logic of the model. Do this by copying the file `prog_model_template.py` and replacing the functions with the logic specific to your model. 

For this example, we will model the throwing of an object directly into the air in a vacuum. This is not a particularly interesting problem, but it is simple and illustrates the basic methods of a PrognosticsModel.

The model is illustrated below:


```python
from progpy import PrognosticsModel

class ThrownObject(PrognosticsModel):
    """
    Model that simulates an object thrown directly into the air (vertically) without air resistance
    """

    inputs = [] # no inputs, no way to control
    states = [
        'x', # Vertical position (m) 
        'v'  # Velocity (m/s)
        ]
    outputs = [ # Anything we can measure
        'x' # Position (m)
    ]
    events = [ # Events that can/will occur during simulation
        'falling', # Event- object is falling
        'impact' # Event- object has impacted ground
    ]

    # The Default parameters for any ThrownObject. 
    # Overwritten by passing parameters into constructor as kwargs or by setting model.parameters
    default_parameters = {
        'thrower_height': 1.83, # Height of thrower (m)
        'throwing_speed': 40, # Velocity at which the ball is thrown (m/s)
        'g': -9.81, # Acceleration due to gravity (m/s^2)
        'process_noise': 0.0 # amount of noise in each step
    }    

    # First function: Initialize. This function is used to initialize the first state of the model.
    # This is only needed in cases with complex initialization. 
    # In this case we need it because of max_x
    # If not included, parameters['x0'] is returned as the initial state
    # In this case we do not need input (u) or output (z) to initialize the model, 
    #   so we set them to None, but that's not true for every model.
    # u and z are Input and Output, respectively. 
    # Values can be accessed like a dictionary (e.g., z['x']) using the keys from inputs and outputs, respectively.
    # or they can be accessed using the matrix (i.e., z.matrix)
    def initialize(self, u=None, z=None):
        self.max_x = 0.0
        return self.StateContainer({
            'x': self.parameters['thrower_height'], # initial altitude is height of thrower
            'v': self.parameters['throwing_speed'] 
            })
    
    # Second function: state transition. 
    # State transition can be defined in one of two ways:
    #   1) Discrete models use next_state(x, u, dt) -> x'
    #   2) Continuous models (preferred) use dx(x, u) -> dx/dt
    #
    # In this case we choose the continuous model, so we define dx(x, u)
    # This function defines the first derivative of the state with respect to time, as a function of model configuration (self.parameters), state (x) and input (u).
    # Here input isn't used. But past state and configuration are.
    # 
    # x and u are State and Input, respectively. 
    # Values can be accessed like a dictionary (e.g., x['x']) using the keys from states and inputs, respectively.
    # or they can be accessed using the matrix (i.e., x.matrix)
    def dx(self, x, u):
        return self.StateContainer({
            'x': x['v'],  # dx/dt = v
            'v': self.parameters['g'] # Acceleration of gravity
        })
    # Equivalently, the state transition could have been defined as follows:
    # def next_state(self, x, u, dt):
    #     return self.StateContainer({
    #         'x': x['x'] + x['v']*dt,
    #         'v': x['v'] + self.parameters['g']*dt
    #     })

    # Now, we define the output equation. 
    # This function estimates the output (i.e., measured values) given the system state (x) and system parameters (self.parameters).
    # In this example, we're saying that the state 'x' can be directly measured. 
    # But in most cases output will have to be calculated from state. 
    def output(self, x):
        return self.OutputContainer({
            'x': x['x']
        })

    # Next, we define the event state equation
    # This is the first equation that actually describes the progress of a system towards the events.
    # This function maps system state (x) and system parameters (self.parameters) to event state for each event.
    # Event state is defined as a number between 0 and 1 where 1 signifies no progress towards the event, and 0 signifies the event has occurred.
    # The event keys were defined above (model.events)
    # Here the two event states are as follows:
    #  1) falling: 1 is defined as when the system is moving at the maximum speed (i.e., throwing_speed), and 0 is when velocity is negative (i.e., falling)
    #  2) impact: 1 is defined as the ratio of the current altitude (x) to the maximum altitude (max_x), and 0 is when the current altitude is 0 (i.e., impact)
    def event_state(self, x): 
        self.max_x = max(self.max_x, x['x']) # Maximum altitude
        return {
            'falling': max(x['v']/self.parameters['throwing_speed'],0), # Throwing speed is max speed
            'impact': max(x['x']/self.max_x,0) # Ratio of current altitude to maximum altitude
        }

    # Finally, we define the threshold equation.
    # This is the second equation that describes the progress of a system towards the events.
    # Note: This function is optional. If not defined, threshold_met will be defined as when the event state is 0.
    # However, this implementation is more efficient, so we included it
    # This function maps system state (x) and system parameters (self.parameters) a boolean indicating if the event has been met for each event.
    def threshold_met(self, x):
        return {
            'falling': x['v'] < 0,
            'impact': x['x'] <= 0
        }

```

Now the model can be generated and used like any of the other provided models


```python
m = ThrownObject()

def future_load(t, x=None):
        return m.InputContainer({}) # No loading
event = 'impact'  # Simulate until impact

(times, inputs, states, outputs, event_states) = m.simulate_to_threshold(future_load, events=[event], dt=0.005, save_freq=1)

# Plot results
event_states.plot(ylabel= ['falling', 'impact'], compact= False)
states.plot(ylabel= {'x': "position (m)", 'v': 'velocity (m/s)'}, compact= False)
```

Note: The plots are at the resolution of save_freq (one point per second)

See also `examples.new_model` for more information on building models

## Building a new model - advanced

### Derived Parameters 

Models can also include "derived parameters" (i.e., parameters that are derived from others). These can be set using the param_callbacks property. 

Let's extend the above model to include derived_parameters. Let's say that the throwing_speed was actually a function of thrower_height (i.e., a taller thrower would throw the ball faster). Here's how we would implement that


```python
# Step 1: Define a function for the relationship between thrower_height and throwing_speed.
def update_thrown_speed(params):
    return {
        'throwing_speed': params['thrower_height'] * 21.85
    }  # Assumes thrown_speed is linear function of height

# Step 2: Define the param callbacks
ThrownObject.param_callbacks = {
        'thrower_height': [update_thrown_speed]
    }  # Tell the derived callbacks feature to call this function when thrower_height changes.
```

Note: Usually we would define this method within the class. For this example, we're doing it separately to improve readability. Here's the feature in action


```python
obj = ThrownObject()
print("Default Settings:\n\tthrower_height: {}\n\tthowing_speed: {}".format(obj.parameters['thrower_height'], obj.parameters['throwing_speed']))

# Now let's change the thrower_height
print("changing height...")
obj.parameters['thrower_height'] = 1.75  # Our thrower is 1.75 m tall
print("\nUpdated Settings:\n\tthrower_height: {}\n\tthowing_speed: {}".format(obj.parameters['thrower_height'], obj.parameters['throwing_speed']))
print("Notice how speed changed automatically with height")

# Let's delete the callback so we can use the same model in the future:
ThrownObject.param_callbacks = {}
```

### State Limits

In many cases, the values of the model states have certain physical limits. For example, temperature is limited by absolute zero. In these cases, it is useful to define those limits. In simulation, the defined limits are enforced as the state transitions to prevent the system from reaching an impossible state.

For example, the ThrownObject model can be extended as follows:


```python
from numpy import inf

ThrownObject.state_limits = {
        # object may not go below ground
        'x': (0, inf),

        # object may not exceed the speed of light
        'v': (-299792458, 299792458)
    }
```

Note: like derived parameters, these would typically be defined in class definition, not afterwards. They are defined afterwards in this case to illustrate the feature.

State limits can be applied directly using the apply_limits function. For example:


```python
x = {'x': -5, 'v': 3e8} # Too fast and below the ground
x = obj.apply_limits(x)
print(x)
```

Notice how the state was limited according to the model state limits and a warning was issued. The warning can be suppressed by suppressing ProgModelStateLimitWarning (`warnings.simplefilter('ignore', ProgModelStateLimitWarning)`)

See also examples.derived_params for more information on this feature.

## Parameter Estimation

Let's say we don't know the configuration of the above model. Instead, we have some data. We can use that data to estimate the parameters. 

First, we define the data:


```python
times = [0, 1, 2, 3, 4, 5, 6, 7, 8]
inputs = [{}]*9
outputs = [
    {'x': 1.83},
    {'x': 36.95},
    {'x': 62.36},
    {'x': 77.81},
    {'x': 83.45},
    {'x': 79.28},
    {'x': 65.3},
    {'x': 41.51},
    {'x': 7.91},
]
```

Next, we identify which parameters will be optimized


```python
keys = ['thrower_height', 'throwing_speed']
```

Let's say we have a first guess that the thrower's height is 20m, silly I know


```python
m.parameters['thrower_height'] = 20
```

Here's the state of our estimation with that assumption:


```python
print('Model configuration before')
for key in keys:
    print("-", key, m.parameters[key])
```

We can also calculate the error between simulated and observed data given this incorrect assumption. The calc_error functionality can calculate error using a variety of methods, including Mean Squared Error, which we use here. 


```python
print(' Error: ', m.calc_error(times, inputs, outputs, dt=1e-4, method='MSE'))
```

Wow, that's a large error. 

Let's run the parameter estimation to see if we can do better


```python
m.estimate_params([(times, inputs, outputs)], keys, dt=0.01)

# Print result
print('\nOptimized configuration')
for key in keys:
    print("-", key, m.parameters[key])
print(' Error: ', m.calc_error(times, inputs, outputs, dt=1e-4))
```

Much better!

See also examples.param_est for more details about this feature

## UncertainData - Representing a Distribution

Uncertainty is sometimes present in data used for performing state estimations or making predictions.

In `progpy`, data with uncertainty is represented using classes inheriting from `UncertainData`:
* `progpy.uncertain_data.MultivariateNormalDist` - Data represented by a multivariate normal distribution with mean and covariance matrix
* `progpy.uncertain_data.ScalarData` - Data without uncertainty, a single value
* `progpy.uncertain_data.UnweightedSamples` - Data represented by a set of unweighted samples. Objects of this class can be treated like a list where samples[n] returns the nth sample (Dict)

To begin using `UncertainData`, import the type that best portrays the data. In this simple demonstration, we import the `UnweightedSamples` data type. See <https://nasa.github.io/progpy/api_ref/progpy/UncertainData.html> for full details on the available `UncertainData` types.


```python
from progpy.uncertain_data import UnweightedSamples
```

With `UnweightedSamples` imported, construct an object with samples. This object can be initialized using either a dictionary, list, or model.*Container type from prog_models (e.g., StateContainer). Let's try creating an object using a dictionary. 


```python
samples = UnweightedSamples([{'x': 1, 'v':2}, {'x': 3, 'v':-2}])
print(samples)
```

Given an integer value, addition and subtraction can be performed on the `UncertainData` classes to adjust the distribution by a scalar amount.


```python
samples = samples + 5
print(samples)
samples -= 3
print(samples)
```

We can also sample from any `UncertainData` distribution using the `sample` method. In this case it resamples from the existing samples


```python
print(samples.sample()) # A single sample
print(samples.sample(10)) # 10 samples
```

We can see the keys present using the `.keys()` method:


```python
print(samples.keys())
```

and the data corresponding to a specific key can be retrieved using `.key()`


```python
print(samples.key('x'))
```

Various properties are available to quantify the `UncertainData` distribution


```python
print('mean', samples.mean)
print('median', samples.median)
print('covariance', samples.cov)
print('size', samples.size)
```

These `UncertainData` classes are used throughout the prog_algs package to represent data with uncertainty, as described in the following sections.

## State Estimation

State estimation is the process of estimating the system state given sensor data and a model. Typically, this is done repeatedly as new sensor data is available.

In `progpy` a State Estimator is used to estimate the system state. 

To start, import the needed packages. Here we will import the `BatteryCircuit` model and the `UnscentedKalmanFilter` state estimator. See <https://nasa.github.io/progpy/api_ref/progpy/StateEstimator.html> for more details on the available state estimators.


```python
from progpy.models import BatteryCircuit
from progpy.state_estimators import UnscentedKalmanFilter
```

Next we construct and initialize the model. 

We use the resulting model and initial state to construct the state estimator. 


```python
m = BatteryCircuit()
x0 = m.initialize()

# Turn into a distribution - this represents uncertainty in the initial state
from progpy.uncertain_data import MultivariateNormalDist
from numpy import diag
INITIAL_UNCERT = 0.05  # Uncertainty in initial state (%)
# Construct covariance matrix (making sure each value is positive)
cov = diag([max(abs(INITIAL_UNCERT * value), 1e-9) for value in x0.values()])
x0 = MultivariateNormalDist(x0.keys(), x0.values(), cov)

# Construct State estimator
est = UnscentedKalmanFilter(m, x0)
```

Now we can use the estimator to estimate the system state.


```python
print("Prior State:", est.x.mean)
print('\tSOC: ', m.event_state(est.x.mean)['EOD'])
fig = est.x.plot_scatter(label='prior')

t = 0.1
u = m.InputContainer({'i': 2})
example_measurements = m.OutputContainer({'t': 32.2, 'v': 3.915})
est.estimate(t, u, example_measurements)

print("Posterior State:", est.x.mean)
print('\tSOC: ', m.event_state(est.x.mean)['EOD'])
est.x.plot_scatter(fig= fig, label='posterior')
```

As mentioned previously, this step is typically repeated when there's new data. filt.x may not be accessed every time the estimate is updated, only when it's needed.

## Prediction Example

Prediction is the practice of using a state estimation, a model, and estimates of future loading to predict future states and when an event will occur.

First we will import a predictor. In this case, we will use the MonteCarlo Predictor, but see documentation <https://nasa.github.io/progpy> for a full list of predictors and their configuration parameters.


```python
from progpy.predictors import MonteCarlo
```

Next we initialize it using the model from the above example


```python
mc = MonteCarlo(m)
```

Next, let's define future loading and the first state. The first state is the output of the state estimator, and the future loading scheme is a simple piecewise function


```python
x = est.x  # The state estimate

from progpy.loading import Piecewise
future_load = Piecewise(
    m.InputContainer,
    [600, 900, 1800, 3000],
    {'i': [2, 1, 4, 2, 3]}
)
```

Now let's use the constructed mc predictor to perform a single prediction. Here we're setting dt to 0.25. Note this may take up to a minute


```python
mc_results = mc.predict(x, future_loading, dt=0.25, n_samples=20)
```

The predict function returns predictions of future inputs, states, outputs, and event_states at each save point. For sample-based predictors like the monte carlo, these can be accessed like an array with the format `[sample #][time]` so that `mc_results.states[m][n]` corresponds to the state for sample `m` at time `mc_results.times[m][n]`. Alternately, use the method `snapshot` to get a  single point in time. e.g., 

 state = mc_results.states.snapshot(3)

In this case the state snapshot corresponds to time `mc_results.times[3]`. The snapshot method returns type UncertainData. 

The `predict` method also returns Time of Event (ToE) as a type UncertainData, representing the predicted time of event (for each event predicted), with uncertainty.

Next, let's use the metrics package to analyze the ToE


```python
print("\nEOD Predictions (s):")
print('\tPortion between 3005.2 and 3005.6: ', mc_results.time_of_event.percentage_in_bounds([3005.2, 3005.6]))
print('\tAssuming ground truth 3005.25: ', mc_results.time_of_event.metrics(ground_truth = 3005.25))
from progpy.metrics import prob_success 
print('\tP(Success) if mission ends at 3005.25: ', prob_success(mc_results.time_of_event, 3005.25))
```

These analysis methods applied to ToE can also be applied to anything of type UncertainData (e.g., state snapshot). 

You can also visualize the results in a variety of different ways. For example, state transition


```python
fig = mc_results.states.snapshot(0).plot_scatter(label = "t={:.0f}".format(int(mc_results.times[0])))
for i in range(1, 4):
    index = int(len(mc_results.times)/4*i)
    mc_results.states.snapshot(index).plot_scatter(fig=fig, label = "t={:.0f}".format(mc_results.times[index]))
mc_results.states.snapshot(-1).plot_scatter(fig = fig, label = "t={:.0f}".format(int(mc_results.times[-1])))
```

Or time of event (ToE)


```python
fig = mc_results.time_of_event.plot_hist()
```

Note, for this event, there is only one event (End of Discharge). Many models have multiple events that can be predicted. For these models, ToE for each event is returned and can be analyzed.

Alternately, a specific event (or events) can be specified for prediction. See `examples.predict_specific_event` for more details.

Frequently the prediction step is run periodically, less often than the state estimator step

## Extending - Adding a new state estimator

New state estimators can be created by extending the state_estimator interface. As an example lets use a really dumb state estimator that adds random noise each step - and accepts the state that is closest. 

First thing we need to do is import the StateEstimator parent class


```python
from progpy.state_estimators import StateEstimator
```

Next we select how state will be represented. In this case there's no uncertainty- so we represent it as a scaler. Import the appropriate class


```python
from prog_algs.uncertain_data import ScalarData
```

Now we construct the class, implementing the functions of the state estimator template (`state_estimator_template.py`)


```python
import random 

class BlindlyStumbleEstimator(StateEstimator):
    def __init__(self, model, x0):
        self.m = model
        self.state = x0

    def estimate(self, t, u, z):
        # Generate new candidate state
        x2 = {key : float(value) + 10*(random.random()-0.5) for (key,value) in self.state.items()}

        # Calculate outputs
        z_est = self.m.output(self.state)
        z_est2 = self.m.output(x2)

        # Now score them each by how close they are to the measured z
        z_est_score = sum([abs(z_est[key] - z[key]) for key in self.m.outputs])
        z_est2_score = sum([abs(z_est2[key] - z[key]) for key in self.m.outputs])

        # Now choose the closer one
        if z_est2_score < z_est_score: 
            self.state = x2

    @property
    def x(self):
        return ScalarData(self.state)
```

Great, now let's try it out using the model from earlier. with an initial state of all 0s. It should slowly converge towards the correct state


```python
x0 = {key: 0 for key in m.states}
se = BlindlyStumbleEstimator(m, x0)

for i in range(25):
    u = m.InputContainer({'i': 0})
    z = m.OutputContainer({'t': 18.95, 'v': 4.183})
    se.estimate(i, u, z)
    print(se.x.mean)
    print("\tcorrect: {'tb': 18.95, 'qb': 7856.3254, 'qcp': 0, 'qcs': 0}")
```

## Extending - Adding a new Predictor

Like the example above with StateEstimators, Predictors can be extended by subclassing the Predictor class. Copy `predictor_template.py` as a starting point.

## Conclusion
This is just the basics, there's much more to learn. Please see the documentation at <https://nasa.github.io/progpy/guide> and the examples in the `examples/` folder for more details on how to use the package, including:
* `examples.derived_params` : Example building models with derived parameters.
* `examples.state_limits`: Example building models with limits on state variables.
* `examples.param_est`: Example using the parameter estimation feature 
* `examples.dynamic_step_size`: Example simulating with dynamic (i.e., changing as a function of time or state) step size.
* `examples.events`: Example extending a model to include additional events, such as warning thresholds.
* `examples.generate_surrogate`: Example generating a surrogate model
* `examples.linear_model`: Example using the new Linear Model subclass
* `examples.benchmarking`: Example benchmarking the performance of a Prognostics Model
* `examples.future_loading`: Example with complex future loading schemes
* `examples.new_model`: Example building a new model
* `examples.noise`: Example demonstrating how noise can be added in simulation
* `examples.vectorized`: Example simulating a vectorized model
* `examples.sim`, `examples.sim_battery_eol`, `examples.sim_pump`, `examples.sim_valve`, `examples.sim_powertrain`, `examples.sim_dcmotor_singlephase`: Examples using specific models from `progpy.models`
* `examples.lstm_model`, `examples.full_lstm_model`, and `examples.custom_model`: Examples using data-driven models
* `examples.basic_example` : A basic Example using prog_algs for Prognostics 
* `examples.basic_example_battery` : A basic Example using prog_algs for Prognostics, using the more complex battery model
* `examples.eol_event` : An example where a model has multiple events, but the user is only interested in predicting the time when the first event occurs (whatever it is).
* `examples.measurement_eqn_example` : An example where not every output is measured or measurements are not in the same format as outputs, so a measurement equation is defined to translate between outputs and what's measured. 
* `examples.new_state_estimator_example` : An example of extending StateEstimator to create a new state estimator class
* `examples.playback` : A full example performing prognostics using playback data.
* `examples.predict_specific_event` : An example where the model has multiple events, but the user is only interested in predicting a specific event (or events).

Thank you for trying out this tutorial. Open an issue on github (https://github.com/nasa/progpy/issues) or email Chris Teubert (christopher.a.teubert@nasa.gov) with any questions or issues.

Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
