<center>
<h1>ProgPy Tutorial</h1>
<h3>2024 NASA Software of the Year!</h3>
2024 PHM Society Conference

November, 2024

**Presenter**: Chris Teubert (christopher.a.teubert@nasa.gov)

**In-room help**: Chetan Kulkarni & Rajeev Ghimire

**Online help**: Katelyn Griffith

**Technical Help**: Raise your hand if you need technical help, someone from our team will come around and help you.

**Questions**: Please put questions in the Whova App during the presentation or we will stop at various points to answer questions. 

# Pre-Work

## 1. Download Whova
Please download the Whova App for live Q&A during the session. The Q&A can be found here: https://whova.com/portal/webapp/phm1_202411/Agenda/4229218

## 2. Installing ProgPy
_We recommend installing ProgPy prior to the tutorial_

ProgPy requires a version of Python between 3.7-3.12

The latest stable release of ProgPy is hosted on PyPi. To install via the command line, use the following command: 

`$ pip install progpy`

## 3. Start Jupyter Notebook
Either by cloning the git repo

$ git clone https://github.com/nasa/progpy.git

or using binder: 
https://mybinder.org/v2/gh/nasa/progpy/master?labpath=examples/2024PHMTutorial.ipynb

## 4. Download data
Next, lets download the data we will be using for this tutorial. To do this we will use the datasets subpackage in progpy.


```python
from progpy.datasets import nasa_battery
(desc, data) = nasa_battery.load_data(1)
```

Note, this downloads the battery data from the PCoE datasets:
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ 

# Introduction to ProgPy 
<center><h3> 2024 NASA Software of the Year!!! </h3></center>

NASA’s ProgPy is an open-source python package supporting research and development of prognostics, health management, and predictive maintenance tools. It implements architectures and common functionality of prognostics, supporting researchers and practitioners.

The goal of this tutorial is to instruct users how to use and extend ProgPy. This tutorial will cover how to use a model, including existing models and additional capabilities like parameter estimation and simulation, as well as how to build a new model from scratch. 

The tutorial will begin with an introduction to prognostics and ProgPy using ProgPy's documentation. Please follow along in the [ProgPy Guide](https://nasa.github.io/progpy/guide.html).

### Tutorial Outline

0. The Dataset 
1. Using an existing model
    - Loading a model
    - Model parameters
    - Simulation
    - Prognostics with data
2. Building a new model 
    - State transition 
    - Outputs
    - Events
    - Using the model
    - Parameter estimation
    - Prognostics example
    - Final notes 
3. Advanced Capabilities
 

# The Dataset

Let's prepare the dataset that we will use for this tutorial.


```python
print(desc['description'])
```

The dataset includes 4 different kinds of runs: trickle, step, reference, random walk. We're going to split the dataset into one example for each of the different types for use later.

Let's take a look at the trickle discharge run first.


```python
trickle_dataset = data[0]
print(trickle_dataset)
trickle_dataset.plot(y=['current', 'voltage', 'temperature'], subplots=True, xlabel='Time (sec)')
```

Now let's do the same for a reference discharge run (5).


```python
reference_dataset = data[5]
reference_dataset.plot(y=['current', 'voltage', 'temperature'], subplots=True, xlabel='Time (sec)')
```

Now let's do it for the step runs. Note that this is actually multiple runs that we need to combine. 

relativeTime resets for each "run". So if we're going to use multiple runs together, we need to stitch these times together.


```python
data[7]['absoluteTime'] = data[7]['relativeTime']
for i in range(8, 32):
    data[i]['absoluteTime'] = data[i]['relativeTime'] + data[i-1]['absoluteTime'].iloc[-1]
```

Next, we should combine the data into a single dataset and investigate the results


```python
import pandas as pd
step_dataset = pd.concat(data[7:32], ignore_index=True)
print(step_dataset)
step_dataset.plot(y=['current', 'voltage', 'temperature'], subplots=True, xlabel='Time (sec)')
```

Finally, let's investigate the random walk discharge

Like the step discharge, we need to stitch together the times and concatenate the data


```python
data[35]['absoluteTime'] = data[35]['relativeTime']
for i in range(36, 50):
    data[i]['absoluteTime'] = data[i]['relativeTime'] + data[i-1]['absoluteTime'].iloc[-1]
```


```python
random_walk_dataset = pd.concat(data[35:50], ignore_index=True)
print(random_walk_dataset)
random_walk_dataset.plot(y=['current', 'voltage', 'temperature'], subplots=True, xlabel='Time (sec)')
```

Now the data is ready for this tutorial, let's dive into it.

# Using an existing Model

The first component of ProgPy are the **Prognostics Models**. Models describe the behavior of the system of interest and how the state of the system evolves with use. ProgPy includes capability for prognostics models to be [physics-based](https://nasa.github.io/progpy/glossary.html#term-physics-based-model) or [data-driven](https://nasa.github.io/progpy/glossary.html#term-data-driven-model).

All prognostics models have the same [format](https://nasa.github.io/progpy/prog_models_guide.html#progpy-prognostic-model-format) within ProgPy. The architecture requires definition of model inputs, states, outputs, and events which come together to create a system model.

ProgPy includes a collection of [included models](https://nasa.github.io/progpy/api_ref/progpy/IncludedModels.html#included-models) which can be accessed through the `progpy.models` package.


### Loading a Model

To illustrate how to use a built-in model, let's use the [Battery Electrochemistry model](https://nasa.github.io/progpy/api_ref/progpy/IncludedModels.html#:~:text=class%20progpy.models.BatteryElectroChemEOD(**kwargs)). This model predicts the end-of-discharge of a Lithium-ion battery based on a set of differential equations that describe the electrochemistry of the system [Daigle et al. 2013](https://papers.phmsociety.org/index.php/phmconf/article/view/2252).



First, import the model from the `progpy.models` package.


```python
from progpy.models import BatteryElectroChemEOD
```

Next, let's create a new battery using the default parameters:


```python
batt = BatteryElectroChemEOD()
```

### Model parameters

Model parameters describe the specific system the model will simulate. For the Electrochemistry model, the default model parameters are for 18650-type Li-ion battery cells. All parameters can be accessed through `batt.parameters`. Let's print out all of the parameters, followed by the specific parameter for Ohmic drop, denoted as `Ro` in this model.


```python
print(batt.parameters)
print(batt['Ro'])
```

Parameter values can be configured in various ways. Parameter values can be passed into the constructor as keyword arguments when the model is first instantiated or can be set afterwards. Let's change two parameters to be more specific to our battery use-case:


```python
batt['Ro'] = 0.15
batt['qMobile'] = 7750
```

In addition to setting model parameter values by hand, ProgPy includes a [parameter estimation](https://nasa.github.io/progpy/prog_models_guide.html#parameter-estimation:~:text=examples.future_loading-,Parameter%20Estimation,-%23) functionality that tunes the parameters of a general model to match the behavior of a specific system. In ProgPy, the `progpy.PrognosticsModel.estimate_params()` method tunes model parameters so that the model provides a good fit to observed data. In the case of the Electrochemistry model, for example, parameter estimation would take the general battery model and configure it so that it more accurately describes a specific battery. The ProgPy documentation includes a [detailed example](https://nasa.github.io/progpy/prog_models_guide.html#parameter-estimation:~:text=See%20the%20example%20below%20for%20more%20details) on how to do parameter estimation.

### Simulation

Once a model has been created, the next step is to simulate it's evolution throughout time. Simulation is the foundation of prediction, but unlike full prediction, simulation does not include uncertainty in the state and other product (e.g., [output](https://nasa.github.io/progpy/glossary.html#term-output)) representation.

*Future Loading*

Most prognostics models have some sort of [input](https://nasa.github.io/progpy/glossary.html#term-input), i.e. a control or load applied to the system that impacts the system state and outputs. For example, for a battery, the current drawn from the battery is the applied load, or input. In this case, to simulate the system, we must define a `future_loading` function that describes how the system will be loaded, or used, throughout time. (Note that not all systems have applied load, e.g. [ThrownObject](https://nasa.github.io/progpy/api_ref/progpy/IncludedModels.html?highlight=thrownobject#progpy.models.ThrownObject), and no `future_loading` is required in these cases.)

ProgPy includes pre-defined [loading functions](https://nasa.github.io/progpy/api_ref/progpy/Loading.html?highlight=progpy%20loading) in `progpy.loading`. Here, we'll implement the built-in piecewise loading functionality.


```python
from progpy.loading import Piecewise

future_loading = Piecewise(
        InputContainer=batt.InputContainer,
        times=[600, 900, 1800, 3000],
        values={'i': [2, 1, 4, 2, 3]})
```

*Simulate to Threshold*

With this in mind, we're ready to simulate our model forward in time using ProgPy's [simulation functionality](https://nasa.github.io/progpy/prog_models_guide.html#simulation).

Physical systems frequently have one or more failure modes, and there's often a need to predict the progress towards these events and the eventual failure of the system. ProgPy generalizes this concept of predicting Remaining Useful Life (RUL) with [events](https://nasa.github.io/progpy/prog_models_guide.html#events) and their corresponding thresholds at which they occur. 


Often, there is interest in simulating a system forward in time until a particular event occurs. ProgPy includes this capability with `simulate_to_threshold()`. 

First, let's take a look at what events exist for the Electrochemistry model.


```python
batt.events
```

The only event in this model is 'EOD' or end-of-discharge. The `progpy.PrognosticsModel.event_state()` method estimates the progress towards the event, with 1 representing no progress towards the event and 0 indicating the event has occurred.  The method `progpy.PrognosticsModel.threshold_met()` defines when the event has happened. In the Electrochemistry model, this occurs when the battery voltage drops below some pre-defined value, which is stored in the parameter `VEOD`. Let's see what this threshold value is.


```python
batt.parameters['VEOD']
```

With these definitions in mind, let's simulate the battery model until threshold for EOD is met. We'll use the same `future_loading` function as above. 


```python
results = batt.simulate_to_threshold(
    future_loading,
    save_freq=10,  # Frequency at which results are saved (s)
    horizon=8000  # Maximum time to simulate (s) - This is a cutoff. The simulation will end at this time, or when a threshold has been met, whichever is first
)
```

Let's visualize the results. Note that the simulation ends when the voltage value hits the VEOD value of 3.0.


```python
fig = results.inputs.plot(ylabel='Current drawn (amps)')
fig = results.event_states.plot(ylabel='Battery State of Charge')
fig = results.outputs.plot(ylabel= {'v': "voltage (V)", 't': 'temperature (°C)'}, compact= False)
```

In addition to simulating to threshold, ProgPy also includes a simpler capability to simulate until a particular time, using `simulate_to()`.

### Prognostics with data



Now that we have a basic simulation of our model, let's make a prediction using the prognostics capabilities within ProgPy. The two basic components of prognostics are [state estimation and prediction](https://nasa.github.io/progpy/prog_algs_guide.html#state-estimation-and-prediction-guide). ProgPy includes functionality to do both.

To implement a prognostics example, we first need data from our system. We'll use the data we've already uploaded and prepped above.

For the battery electrochemistry model, we'll need to use a [state estimator](https://nasa.github.io/progpy/prog_algs_guide.html#state-estimation) because the model state is not directly measureable, i.e. it has hidden states. We'll use a Particle Filter and the `estimate` method. ProgPy also includes a Kalman Filter and an Unscented Kalman Filter.

First, let's load the necessary imports.


```python
import numpy as np
from progpy.state_estimators import ParticleFilter
from progpy.uncertain_data import MultivariateNormalDist
```

State estimators require an initial state. To define this, we'll first initialize the model and then define the initial state as a distribution of possible states around this using a multi-variate normal distribution. 


```python
initial_state = batt.initialize() # Initialize model
# Define distribution around initial state
x_guess = MultivariateNormalDist(
    labels=initial_state.keys(),
    mean=initial_state.values(),
    covar=np.diag([max(1e-9, abs(x)) for x in initial_state.values()])
)
```

With our initial distribution defined, we can now instantiate the state estimator.


```python
pf = ParticleFilter(batt, x_guess)
```

With this, we're ready to run the State Estimator. To illustrate how state estimation works, let's estimate one step forward in time. First, we'll extract the measurement at this time. 


```python
# Define time step based on data
dt = random_walk_dataset['absoluteTime'][1] - random_walk_dataset['absoluteTime'][0]

# Data at time point
z = {'t': random_walk_dataset['temperature'][1], 'v': random_walk_dataset['voltage'][1]}
```

Next, we'll estimate the new state by calling the `estimate` method. 


```python
# Extract input current from data 
i = {'i': random_walk_dataset['current'][1]}

# Estimate the new state
pf.estimate(dt, i, z)
x_est = pf.x.mean
```

Finally, let's look at the difference between the estimated state and the true measurement. In the following plots, blue circles represent the initial distribution and orange circles represent the estimated result. The orange circles are more refined and give a better estimate, highlighting the usefulness of the state estimator.


```python
fig = x_guess.plot_scatter(label='initial')
fig = pf.x.plot_scatter(fig=fig, label='update')
```

Now that we know how to do state estimation, the next key component of prognostics is [prediction](https://nasa.github.io/progpy/prog_algs_guide.html#prediction). ProgPy includes multiple predictors, and we'll implement a [Monte Carlo](https://nasa.github.io/progpy/api_ref/progpy/Predictor.html?highlight=monte%20carlo#included-predictors) predictor here. Let's load the necessary imports. 


```python
from progpy.predictors import MonteCarlo
```

Next, a key factor in modeling any real-world application is noise. See the ProgPy [noise documentation](https://nasa.github.io/progpy/prog_models_guide.html#noise) for a detailed description of different types of noise and how to include it in the ProgPy architecture. Here, let's add some process and measurement noise into our system, to capture any uncertainties. 


```python
PROCESS_NOISE = 2e-4           # Percentage process noise
MEASUREMENT_NOISE = 1e-4        # Percentage measurement noise

# Apply process noise to state
batt.parameters['process_noise'] = {key: PROCESS_NOISE * value for key, value in initial_state.items()}

# Apply measurement noise to output
z0 = batt.output(initial_state)
batt.parameters['measurement_noise'] = {key: MEASUREMENT_NOISE * value for key, value in z0.items()}
```

Next, let's set up our predictor. 


```python
mc = MonteCarlo(batt)
```

To perform the prediction, we need to specify a few things, including the number of samples we want to use for the prediction, the step size for the prediction, and the prediction horizon (i.e., the time value to predict to).


```python
NUM_SAMPLES = 100
STEP_SIZE = 1
PREDICTION_HORIZON = random_walk_dataset['absoluteTime'].iloc[-1] 
```

We also need to define a future loading function based on the load in the dataset we are using. Let's extract the necessary information and define a function.


```python
# Extract time and outputs from data
times_rw = random_walk_dataset['absoluteTime']
outputs_rw = [{'v': elem[1]['voltage']} for elem in random_walk_dataset.iterrows()]

# Define function
import numpy as np
def future_load_rw(t, x=None):
    current = np.interp(t, times_rw, random_walk_dataset['current'])
    return {'i': current}
```


```python
# Adjust voltage threshold 
batt.parameters['VEOD'] = 3.3
```

With this, we are ready to predict. 


```python
mc_results = mc.predict(initial_state, future_loading_eqn=future_load_rw, n_samples=NUM_SAMPLES, dt=STEP_SIZE, save_freq = 10, horizon=PREDICTION_HORIZON, constant_noise=True)
```

Let's visualize the results. We'll plot 1) the data (in orange), 2) the predicted mean value (blue), 3) the individual predictions to show uncertainty (grey).


```python
import matplotlib.pyplot as plt
for z in mc_results.outputs:
    plt.plot(z.times, [z_i['v'] for z_i in z], 'grey', linewidth=0.5)
plt.plot(z.times, [z_i['v'] for z_i in mc_results.outputs[-1]], 'grey', linewidth=0.5, label='MC Samples')
fig = plt.plot(mc_results.times, [z['v'] for z in mc_results.outputs.mean], label='mean prediction')
fig = plt.plot(random_walk_dataset['absoluteTime'], random_walk_dataset['voltage'], label='ground truth')
plt.plot([0, PREDICTION_HORIZON], [batt['VEOD'], batt['VEOD']], color='red', label='EOD Threshold')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Voltage (volts)')
```


```python
mc_results.outputs.mean
```

<center>~~~STOP FOR QUESTIONS~~~</center>

Now that we understand the basics of state estimation and prediction, as well as how to implement these concepts within ProgPy, we are ready to do a full prognostics example. We'll use the state estimator and predictor we created above.

First, let's set a few values we'll use in the simulation.


```python
# Constant values
NUM_SAMPLES = 50
PREDICTION_UPDATE_FREQ = 50     # Number of steps between prediction updates
```

Next, let's initialize a data structure for storing the results, using the following built-in class:


```python
from progpy.predictors import ToEPredictionProfile
profile = ToEPredictionProfile()
```

Now we'll perform the prognostics. We'll loop through time, estimating the state at each time step, and making a prediction at the `PREDICTION_UPDATE_FREQ`.

For the sake of this tutorial and the data we're using, we need to change the default voltage threshold value. By changing this, we'll make the simulation run faster for our in-person demo, and ensure that samples reach EOD before the simulation is over. In practice, this value should be chosen based on the specific use-case you're considering. 


```python
# Loop through time
for ind in range(3, random_walk_dataset.shape[0]):
    # Extract data
    t = random_walk_dataset['absoluteTime'][ind]
    i = {'i': random_walk_dataset['current'][ind]}
    z = {'t': random_walk_dataset['temperature'][ind], 'v': random_walk_dataset['voltage'][ind]}

    # Perform state estimation 
    pf.estimate(t, i, z)
    eod = batt.event_state(pf.x.mean)['EOD']
    print("  - Event State: ", eod)

    # Prediction step (at specified frequency)
    if (ind%PREDICTION_UPDATE_FREQ == 0):
        # Perform prediction
        mc_results = mc.predict(pf.x, future_load_rw, t0 = t, n_samples=NUM_SAMPLES, dt=1, horizon=PREDICTION_HORIZON, const_load=True)
        
        # Calculate metrics and print
        metrics = mc_results.time_of_event.metrics()
        print('  - ToE: {} (sigma: {})'.format(metrics['EOD']['mean'], metrics['EOD']['std']))

        # Save results
        profile.add_prediction(t, mc_results.time_of_event)
```

With our prognostics results, we can now calculate some metrics to analyze the accuracy. 

First, we need to define what the ground truth value is for end-of-discharge.


```python
GROUND_TRUTH = {'EOD': 1600} 
```

We'll start by calculating the cumulative relative accuracy given the ground truth value. 


```python
cra = profile.cumulative_relative_accuracy(GROUND_TRUTH)
print(f"Cumulative Relative Accuracy for 'EOD': {cra['EOD']}")
```

We'll also generate some plots of the results.


```python
ALPHA = 0.05
playback_plots = profile.plot(GROUND_TRUTH, ALPHA, True)
```

### Data-driven Capabilities and Surrogate Models

In addition to the physics-based modeling functionalities described so far, ProgPy also includes a [framework for implementing data-driven models](https://nasa.github.io/progpy/api_ref/progpy/DataModel.html?highlight=surrogate#datamodel). Included methodologies are [Dynamic Mode Decomposition](https://nasa.github.io/progpy/api_ref/progpy/DataModel.html?highlight=surrogate#dmdmodel), [LSTM](https://nasa.github.io/progpy/api_ref/progpy/DataModel.html?highlight=surrogate#lstmstatetransitionmodel), and [Polynomial Chaos Expansion](https://nasa.github.io/progpy/api_ref/progpy/DataModel.html?highlight=surrogate#polynomialchaosexpansion). This data-driven architecture also includes [surrogate models](https://nasa.github.io/progpy/api_ref/progpy/DataModel.html?highlight=surrogate#from-another-prognosticsmodel-i-e-surrogate) which can be used to create models that approximate the original/higher-fidelity models, generally resulting in a less accurate model that is more computationally efficient. 



# Building a new model

The last section described how to use a prognostics model distributed with ProgPy (BatteryElectroChemEOD). However, in many cases a model doesn't yet exist for the system being targeted. In those cases, a new model must be built to describe the behavior and degradation of the system.

In this section we will create a new model from scratch, specifically a new physics-based model. ProgPy also includes tools for training data-driven models (see data-driven tab, here: https://nasa.github.io/progpy/prog_models_guide.html#state-transition-models), but that is not the approach we will be demonstrating today.

Physics-based state transition models that cannot be described linearly are constructed by subclassing [progpy.PrognosticsModel](https://nasa.github.io/progpy/api_ref/prog_models/PrognosticModel.html#prog_models.PrognosticsModel). To demonstrate this, we'll create a new model class that inherits from this class. Once constructed in this way, the analysis and simulation tools for PrognosticsModels will work on the new model.
https://nasa.github.io/progpy/prog_models_guide.html#state-transition-models

We will again be using the battery as a target, creating an alternative to the battery model introduced in the previous section.
We will be implementing the simplified battery model introduced by Gina Sierra, et. al. (https://www.sciencedirect.com/science/article/pii/S0951832018301406).

First, we import the PrognosticsModel class. This is the parent class for all ProgPy Models


```python
from progpy import PrognosticsModel
```

## State Transition
The first step to creating a physics-based model is implementing state transition. From the paper we see one state (SOC) and one state transition equation:

$SOC(k+1) = SOC(k) - P(k)*\Delta t * E_{crit}^{-1} + w_2(k)$

where $k$ is discrete time. The $w$ term is process noise. This can be omitted, since it's handled by ProgPy. 

In this equation we see one input ($P$, power). Note that the previous battery model uses current, where this uses power.

Armed with this information we can start defining our model. First, we start by declaring our inputs and states:


```python
class SimplifiedEquivilantCircuit(PrognosticsModel):
    inputs = ['P']
    states = ['SOC']
```

Next we define parameters. In this case the parameters are the initial SOC state (1) and the E_crit (Internal Total Energy). We get the value for $E_{crit}$ from the paper.

**Note: wont actually subclass in practice, but it's used to break apart model definition into chunks**


```python
class SimplifiedEquivilantCircuit(SimplifiedEquivilantCircuit):
    default_parameters = {
        'E_crit': 202426.858,  # Internal Total Energy
        'x0': {
            'SOC': 1,  # State of Charge
        }
    }
```

We know that SOC will always be between 0 and 1, so we can specify that explicitly.


```python
class SimplifiedEquivilantCircuit(SimplifiedEquivilantCircuit):
    state_limits = {
        'SOC': (0.0, 1.0),
    }
```

Next, we define the state transition equation. There are two methods for doing this: *dx* (for continuous) and *next_state* (for discrete). Today we're using the $dx$ function. This was selected because the model is continuous.


```python
class SimplifiedEquivilantCircuit(SimplifiedEquivilantCircuit):
    def dx(self, x, u):
        return self.StateContainer({'SOC': -u['P']/self['E_crit']})
```

## Outputs

Now that state transition is defined, the next step is to define the outputs of the function. From the paper we have the following output equations:

$v(k) = v_{oc}(k) - i(k) * R_{int} + \eta (k)$

where

$v_{oc}(k) = v_L - \lambda ^ {\gamma * SOC(k)} - \mu * e ^ {-\beta * \sqrt{SOC(k)}}$

and

$i(k) = \frac{v_{oc}(k) - \sqrt{v_{oc}(k)^2 - 4 * R_{int} * P(k)}}{2 * R_{int}(k)}$

There is one output here (v, voltage), the same one input (P, Power), and a few lumped parameters: $v_L$, $\lambda$, $\gamma$, $\mu$, $\beta$, and $R_{int}$. The default parameters are found in the paper.

Note that $\eta$ is the measurement noise, which progpy handles, so that's omitted from the equation below.

Note 2: There is a typo in the paper where the sign of the second term in the $v_{oc}$ term. It should be negative (like above), but is reported as positive in the paper.

We can update the definition of the model to include this output and parameters.


```python
class SimplifiedEquivilantCircuit(SimplifiedEquivilantCircuit):
    outputs = ['v']

    default_parameters = {
        'E_crit': 202426.858,
        'v_L': 11.148,
        'lambda': 0.046,
        'gamma': 3.355,
        'mu': 2.759,
        'beta': 8.482,
        'R_int': 0.027,

        'x0': {
            'SOC': 1,
        }
    }
```

Note that the input ($P(k)$) is also used in the output equation, that means it's part of the state of the system. So we will update the states to include this.

**NOTE: WE CHANGE TO next_state.** Above, we define state transition with ProgPy's `dx` method because the model was continuous. Here, with the addition of power, the model becomes discrete, so we must now use ProgPy's `next_state` method to define state transition. 


```python
class SimplifiedEquivilantCircuit(SimplifiedEquivilantCircuit):
    states = ['SOC', 'P']

    def next_state(self, x, u, dt):
        x['SOC'] = x['SOC'] - u['P'] * dt / self['E_crit']
        x['P'] = u['P']

        return x
    
```

Adding a default P state as well:


```python
class SimplifiedEquivilantCircuit(SimplifiedEquivilantCircuit):
    default_parameters = {
        'E_crit': 202426.858,
        'v_L': 11.148,
        'lambda': 0.046,
        'gamma': 3.355,
        'mu': 2.759,
        'beta': 8.482,
        'R_int': 0.027,

        'x0': {
            'SOC': 1,
            'P': 0.01  # Added P
        }
    }
```

Finally, we're ready to define the output equations (defined above).


```python
import math
class SimplifiedEquivilantCircuit(SimplifiedEquivilantCircuit):
    def output(self, x):
        v_oc = self['v_L'] - self['lambda']**(self['gamma']*x['SOC']) - self['mu'] * math.exp(-self['beta']* math.sqrt(x['SOC']))
        i = (v_oc - math.sqrt(v_oc**2 - 4 * self['R_int'] * x['P']))/(2 * self['R_int'])
        v = v_oc - i * self['R_int']
        return self.OutputContainer({
            'v': v})
```

## Events
Finally we can define events. This is an easy case because our event state (SOC) is part of the model state. So we will simply define a single event (EOD: End of Discharge), where SOC is progress towards that event.


```python
class SimplifiedEquivilantCircuit(SimplifiedEquivilantCircuit):
    events = ['EOD']
```

Then for our event state, we simply extract the relevant state


```python
class SimplifiedEquivilantCircuit(SimplifiedEquivilantCircuit):
    def event_state(self, x):
        return {'EOD': x['SOC']}
```

The threshold of the event is defined as the state where the event state (EOD) is 0.

That's it. We've now defined a complete model. Now it's ready to be used for state estimation or prognostics, like any model distributed with ProgPy

## Using the Model

First step to using the model is initializing an instance of it.


```python
m = SimplifiedEquivilantCircuit()
```

To demonstrate/test this model, we will start by simulating with a constant load


```python
def future_load(t, x=None):
    return {'P': 165}
results = m.simulate_to_threshold(future_load, dt=1, save_freq=1)
```


```python
fig = results.event_states.plot()
```


```python
fig = results.outputs.plot()
```

Everything seems to be working well here. Now let's test how well it fits the random walk dataset. First let's prepare the data and future load equation. Note that this future load uses power instead of current (which the last one used)


```python
times_rw = random_walk_dataset['absoluteTime']
inputs_rw = [elem[1]['voltage'] * elem[1]['current'] for elem in random_walk_dataset.iterrows()]
outputs_rw = [{'v': elem[1]['voltage']} for elem in random_walk_dataset.iterrows()]

import numpy as np
def future_load_rw(t, x=None):
    power = np.interp(t, times_rw, inputs_rw)
    return {'P': power}
```

We can simulate using that future load equation


```python
result = m.simulate_to(random_walk_dataset['absoluteTime'].iloc[-1], future_load_rw, dt=1, save_freq=100)
```

Now let's take a look at the result, comparing it to the ground truth


```python
from matplotlib import pyplot as plt
plt.figure()
plt.plot(times_rw, [z for z in random_walk_dataset['voltage']])
plt.plot(result.times, [z['v'] for z in result.outputs])
plt.xlabel('Time (sec)')
plt.ylabel('Voltage (volts)')
fig = result.event_states.plot()
```

This is a terrible fit. Clearly the battery model isn't properly configured for this specific battery. Reading through the paper we see that the default parameters are for a larger battery pouch present in a UAV, much larger than the 18650 battery that produced our dataset

To correct this, we need to estimate the model parameters.

## Parameter Estimation

Parameter estimation could be a tutorial on its own. Sometimes it can be considered more of an art than a science.

Parameter Estimation the process of estimating the parameters for a model. This is done using a mixture of data, knowledge (e.g., from system specs), and intuition. For large, complex models, it can be VERY difficult and computationall expensive. Fortunately, in this case we have a relatively simple model.

See: https://nasa.github.io/progpy/prog_models_guide.html#parameter-estimation

First, let's take a look at the parameter space


```python
m.parameters
```

This is a very simple model. There are really only 7 parameters to set (assuming initial SOC is always 1).

We can start with setting a few parameters we know. We know that $v_L$ is about 4.2 (from the battery), we expect that the battery internal resistance is the same as that in the electrochemistry model, and we know that the capacity of this battery is significantly smaller.


```python
m['v_L'] = 4.2 # We know this
m['R_int'] = batt['Ro']
m['E_crit'] /= 4 # Battery capacity is much smaller
```

Now let's take a look at the model fit again and see where that got us.


```python
result_guess = m.simulate_to(random_walk_dataset['absoluteTime'].iloc[-1], future_load_rw, dt=1, save_freq=5)
plt.plot(times_rw, [z for z in random_walk_dataset['voltage']])
plt.plot(result_guess.times, [z['v'] for z in result_guess.outputs])
plt.xlabel('Time (sec)')
plt.ylabel('Voltage (volts)')
```

Much better, but not there yet. Next, we need to use the parameter estimation feature to estimate the parameters further. First let's prepare some data. We'll use the trickle, reference, and step datasets for this. These are close enough temporally that we can expect aging effects to be minimal.

**NOTE: It is important to use a different dataset to estimate parameters as to test**


```python
times_trickle = trickle_dataset['relativeTime']
inputs_trickle = [{'P': elem[1]['voltage'] * elem[1]['current']} for elem in trickle_dataset.iterrows()]
outputs_trickle = [{'v': elem[1]['voltage']} for elem in trickle_dataset.iterrows()]

times_ref = reference_dataset['relativeTime']
inputs_ref = [{'P': elem[1]['voltage'] * elem[1]['current']} for elem in reference_dataset.iterrows()]
outputs_ref = [{'v': elem[1]['voltage']} for elem in reference_dataset.iterrows()]

times_step = step_dataset['relativeTime']
inputs_step = [{'P': elem[1]['voltage'] * elem[1]['current']} for elem in step_dataset.iterrows()]
outputs_step = [{'v': elem[1]['voltage']} for elem in step_dataset.iterrows()]
```

Now let's print the keys and the error beforehand for reference. The error here is what is used to estimate parameters.


```python
inputs_reformatted_rw = [{'P': elem[1]['voltage'] * elem[1]['current']} for elem in random_walk_dataset.iterrows()]
all_keys = ['v_L', 'R_int', 'lambda', 'gamma', 'mu', 'beta', 'E_crit']
print('Model configuration')
for key in all_keys:
    print("-", key, m[key])
error_guess = m.calc_error(times=times_rw.to_list(), inputs=inputs_reformatted_rw, outputs=outputs_rw)
print('Error: ', error_guess)
```

Next, lets set the bounds on each of the parameters.

For $v_L$ and $R_{int}$, we're defining some small bounds because we have an idea of what they might be. For the others we are saying it's between 0.1 and 10x the default battery. We also are adding a constraint that E_crit must be smaller than the default, since we know it's a smaller battery.


```python
bounds= {
    'v_L': (3.75, 4.5),
    'R_int': (batt['Ro']*0.5, batt['Ro']*2.5),
    'lambda': (0.046/10, 0.046*10),
    'gamma': (3.355/10, 3.355*10),
    'mu': (2.759/10, 2.759*10),
    'beta': (8.482/10, 8.482*10),
    'E_crit': (202426.858/10, 202426.858) # (smaller than default)
}
```

Finally, we'll estimate the parameters. See [Param Estimation](https://nasa.github.io/progpy/prog_models_guide.html#parameter-estimation) for more details.

We can throw all of the data into estimate parameters, but that will take a LONG time to run, and is prone to errors (e.g., getting stuck in local minima). So, for this example we split characterization into parts.

First we try to capture the base voltage ($v_L$). If we look at the equation above, $v_L$ is the only term that is not a function of either SOC or Power. So, for this estimation we use the trickle dataset, where Power draw is the lowest. We only use the first section where SOC can be assumed to be about 1.


```python
keys = ['v_L']
m.estimate_params(times=trickle_dataset['relativeTime'].iloc[:10].to_list(), inputs=inputs_trickle[:10], outputs=outputs_trickle[:10], keys=keys, dt=1, bounds=bounds)
```

Let's take a look at what that got us:


```python
print('Model configuration')
for key in all_keys:
    print("-", key, m[key])
error_fit1 = m.calc_error(times=times_rw.to_list(), inputs=inputs_reformatted_rw, outputs=outputs_rw)
print(f'Error: {error_guess}->{error_fit1} ({error_fit1-error_guess})')

result_fit1 = m.simulate_to(random_walk_dataset['absoluteTime'].iloc[-1], future_load_rw, dt=1, save_freq=5)
plt.plot(times_rw, [z for z in random_walk_dataset['voltage']], label='ground truth')
plt.plot(result_guess.times, [z['v'] for z in result_guess.outputs], label='guess')
plt.plot(result_fit1.times, [z['v'] for z in result_fit1.outputs], label='fit1')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Voltage (volts)')

plt.figure()
plt.plot([0, 1], [error_guess, error_fit1])
plt.xlabel('Parameter Estimation Run')
plt.ylabel('Error')
plt.ylim((0, 0.25))
```

A tiny bit closer, but not significant. Our initial guess (from the packaging) must have been pretty good.

The next step is to estimate the effect of current on the battery. The Parameter $R_{int}$ (internal resistance) effects this. To estimate $R_{int}$ we will use 2 runs where power is not minimal (ref and step runs). Again, we will use only the first couple steps so EOL can be assumed to be 1.


```python
keys = ['R_int']
m.estimate_params(times=[times_ref.iloc[:5].to_list(), times_step.iloc[:5].to_list()], inputs=[inputs_ref[:5], inputs_step[:5]], outputs=[outputs_ref[:5], outputs_step[:5]], keys=keys, dt=1, bounds=bounds)
```

Again, let's look at what that got us


```python
print('Model configuration')
for key in all_keys:
    print("-", key, m[key])
error_fit2 = m.calc_error(times=times_rw.to_list(), inputs=inputs_reformatted_rw, outputs=outputs_rw)
print(f'Error: {error_fit1}->{error_fit2} ({error_fit2-error_fit1})')

result_fit2 = m.simulate_to(random_walk_dataset['absoluteTime'].iloc[-1], future_load_rw, dt=1, save_freq=5)
plt.plot(times_rw, [z for z in random_walk_dataset['voltage']], label='ground truth')
plt.plot(result_guess.times, [z['v'] for z in result_guess.outputs], label='guess')
plt.plot(result_fit1.times, [z['v'] for z in result_fit1.outputs], label='fit1')
plt.plot(result_fit2.times, [z['v'] for z in result_fit2.outputs], label='fit2')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Voltage (volts)')

plt.figure()
plt.plot([0, 1, 2], [error_guess, error_fit1, error_fit2])
plt.xlabel('Parameter Estimation Run')
plt.ylabel('Error')
plt.ylim((0, 0.25))
```

Much better, but not there yet! Finally we need to estimate the effects of SOC on battery performance. This involves all of the rest of the parameters. For this we will use all the rest of the parameters. We will not be using the entire reference curve to capture a full discharge.

Note we're using the error_method MAX_E, instead of the default MSE. This results in parameters that better estimate the end of the discharge curve and is recommended when estimating parameters that are combined with the event state.


```python
keys = ['lambda', 'gamma', 'mu', 'beta', 'E_crit']
m.estimate_params(times=times_ref.to_list(), inputs=inputs_ref, outputs=outputs_ref, keys=keys, dt=1, bounds=bounds, error_method="MAX_E")
```

Let's see what that got us


```python
print('Model configuration')
for key in all_keys:
    print("-", key, m[key])
error_fit3 = m.calc_error(times=times_rw.to_list(), inputs=inputs_reformatted_rw, outputs=outputs_rw)
print(f'Error: {error_fit2}->{error_fit3} ({error_fit3-error_fit2})')

result_fit3 = m.simulate_to(random_walk_dataset['absoluteTime'].iloc[-1], future_load_rw, dt=1, save_freq=5)
plt.plot(times_rw, [z for z in random_walk_dataset['voltage']], label='ground truth')
plt.plot(result_guess.times, [z['v'] for z in result_guess.outputs], label='guess')
plt.plot(result_fit1.times, [z['v'] for z in result_fit1.outputs], label='fit1')
plt.plot(result_fit2.times, [z['v'] for z in result_fit2.outputs], label='fit2')
plt.plot(result_fit3.times, [z['v'] for z in result_fit3.outputs], label='fit3')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Voltage (volts)')

plt.figure()
plt.plot([0, 1, 2, 3], [error_guess, error_fit1, error_fit2, error_fit3])
plt.xlabel('Parameter Estimation Run')
plt.ylabel('Error')
plt.ylim((0, 0.25))
```

This is even better. Now we have an "ok" estimate, ~150 mV (for the sake of a demo). The estimate could be refined further by setting a lower tolerance (tol parameter), or repeating the 4 parameter estimation steps, above. Talk to Chetan Kulkarni (chetan.s.kulkarni@nasa.gov) with questions on this.

## Prognostics Example
Let's repeat the above example that uses BatteryElectroChemEOD with the same data, so we can compare the results.

This does require an extension of the SimplifiedEquivilantCircuit model. 

In BatteryElectroChemEOD, EOD is defined as when voltage passes below some threshold (VEOD). This is frequently called "functional EOD", because after this point the battery can no longer perform its function.

For SimplifiedEquivilantCircuit, EOD is defined as the point where there is no charge, far after functional EOD. To compare the two, we define a new event: "Low V", for when voltage hits a specific threshold (VEOD).


```python
params = m.parameters # Save learned parameters
class SimplifiedEquivilantCircuit(SimplifiedEquivilantCircuit):
    events = ['EOD', 'Low V']

    def event_state(self, x):
        return {
            'EOD': x['SOC'],
            'Low V': (self.output(x)['v'] - self['VEOD'])/(self['v_L'] - self['VEOD'])
        }

SimplifiedEquivilantCircuit.default_parameters['VEOD'] = batt['VEOD']
m = SimplifiedEquivilantCircuit()
m.parameters.update(params) # update with saved parameters
```

We can then initialize the state distribution. Here we define a distribution with significant noise around SOC and no noise around the power (which is overwritten each step anyway).


```python
initial_state = m.initialize()
x_guess = MultivariateNormalDist(initial_state.keys(), initial_state.values(), np.diag([0.1, 1e-99])) # Define distribution around initial state
```

Now we can construct the Particle Filter with this guess


```python
pf = ParticleFilter(m, x_guess)
fig = pf.x.plot_scatter()
```

Finally we define the process and measurement noise and initialize the predictor.


```python
m.parameters['process_noise'] = {
    'SOC': 5e-5,
    'P': 5e-3}
m.parameters['measurement_noise'] = {
    'v': 0.2
}
m.parameters['process_noise_dist'] = 'normal'
mc = MonteCarlo(m, constant_noise=True)
```

Now let's lake a look at a single prediction using this setup, and plot the results


```python
mc_results = mc.predict(initial_state, future_loading_eqn=future_load_rw, n_samples=NUM_SAMPLES, dt=STEP_SIZE, save_freq=10, horizon=PREDICTION_HORIZON, const_load=True)

for z in mc_results.outputs:
    plt.plot(z.times, [z_i['v'] for z_i in z], 'grey', linewidth=0.5)
fig = plt.plot(mc_results.times, [z['v'] for z in mc_results.outputs.mean], label='mean prediction')
fig = plt.plot(random_walk_dataset['absoluteTime'], random_walk_dataset['voltage'], label='ground truth')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Voltage (volts)')
```

Pretty good, now let's repeat the example from earlier


```python
# Loop through time
simplified_profile = ToEPredictionProfile()
for ind in range(3, random_walk_dataset.shape[0]):

    # Extract data
    t = random_walk_dataset['absoluteTime'][ind]
    i = {'P': random_walk_dataset['current'][ind]*random_walk_dataset['voltage'][ind]}
    z = {'v': random_walk_dataset['voltage'][ind]}

    # Perform state estimation 
    pf.estimate(t, i, z)
    eod = m.event_state(pf.x.mean)['Low V']
    print("  - Event State: ", eod)

    # Prediction step (at specified frequency)
    if (ind%PREDICTION_UPDATE_FREQ == 0):
        # Perform prediction
        mc_results = mc.predict(pf.x, future_load_rw, t0=t, n_samples=NUM_SAMPLES, dt=1, horizon=PREDICTION_HORIZON, events='Low V')

        # Save results
        simplified_profile.add_prediction(t, mc_results.time_of_event)
```

Note the runtime.

Finally let's take a look at the results.


```python
ALPHA = 0.05
playback_plots = profile.plot(GROUND_TRUTH, ALPHA, True)
```

## Final Notes on Building New Models
Here we built a brand new model, from scratch, using the information from a paper. We estimated the parameters using real data and compared its performance with the include BatteryElectroChemEOD.

This is one example on how to build a new model, for more details see https://nasa.github.io/progpy/prog_models_guide.html#building-new-models and the 04_New Models.ipynb file. Other model-building topics include:

* Advanced Noise Representation: e.g., Other distributions
* Complex Future Loading Methods: E.g., moving average, loading with uncertainty or functions of state
* Custom Events: e.g., warning thresholds
* Data-driven models
* Derived Parameters: parameters that are functions of other parameters
* Direct Models: Models of state, future_loading -> Time of Event without state transition
* Linear Models
* Optimizations

Note that this model can be extended by changing the parameters ecrit and r to steady states. This will help the model account for the effects of aging, since they will be estimated with each state estimation step.

# Other Advanced Capabilities

* Combination Models: https://nasa.github.io/progpy/prog_models_guide.html#combination-models and 06_Combining_Models
* Dynamic Step Size
* Integration Methods
* Serialization
* prog_server

## Closing

**[Contributing](https://nasa.github.io/progpy/index.html#contributing-and-partnering)**: Thank you for attending this tutorial. ProgPy is a collaborative effort, including NASA and external collaborators. If you're interested in contributing or learning more, reach out at christopher.a.teubert@nasa.gov

**We are looking or interns for this summer- email christopher.a.teubert@nasa.gov for details**
