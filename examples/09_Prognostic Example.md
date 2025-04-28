# Integrated Prognostics Example

This is an integrated example of Prognostics with ProgPy. This example is based on the prognostics example from the Tutorial at the 2024 PHM Society Conference.

## Data preparation
First, we need to download the data we will use in this chapter. To do this we use the datasets subpackage in progpy.


```python
from progpy.datasets import nasa_battery
(desc, data) = nasa_battery.load_data(1)
```

Note, this downloads the battery data from the PCoE datasets:
https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ 

Let's prepare the dataset.


```python
print(desc['description'])
```

The dataset includes 4 different kinds of runs: trickle, step, reference, random walk. For this example we will use the trickle dataset. 

The dataset includes 4 columns: relativeTime, current, voltage, and temperature. relativeTime is the time in a specific "run" (i.e., with one current draw). To use the random walk dataset, we need to concatenate multiple runs. To support this, we add a new column, absoluteTime, which shows time in the dataset (instead of run).


```python
data[35]['absoluteTime'] = data[35]['relativeTime']
for i in range(36, 50):
    data[i]['absoluteTime'] = data[i]['relativeTime'] + data[i-1]['absoluteTime'].iloc[-1]
```

Next, we combine the data into a single dataset and investigate the results


```python
random_walk_dataset = pd.concat(data[35:50], ignore_index=True)
print(random_walk_dataset)
random_walk_dataset.plot(y=['current', 'voltage', 'temperature'], subplots=True, xlabel='Time (sec)')
```

Now the data is ready for this tutorial, let's dive into it.

## Setting up for Prognostics

To illustrate how to do prognostics, let's use the [Battery Electrochemistry model](https://nasa.github.io/progpy/api_ref/progpy/IncludedModels.html#:~:text=class%20progpy.models.BatteryElectroChemEOD(**kwargs)). This model predicts the end-of-discharge of a Lithium-ion battery based on a set of differential equations that describe the electrochemistry of the system [Daigle et al. 2013](https://papers.phmsociety.org/index.php/phmconf/article/view/2252).

First, lets setup the model.

### Setup Model


```python
from progpy.models import BatteryElectroChemEOD
batt: BatteryElectroChemEOD = BatteryElectroChemEOD()
```

We will also update the Ro and qMobile parameters to better represent the age of the battery. See 02. Parameter Estimation notebook for examples on how to estimate model parameters. 


```python
batt['Ro'] = 0.15
batt['qMobile'] = 7750
```

The two basic components of prognostics are [state estimation and prediction](https://nasa.github.io/progpy/prog_algs_guide.html#state-estimation-and-prediction-guide). ProgPy includes functionality to do both. See 07. State Estimation and 08. Prediction for examples of this.

First, let's setup our state estimator
### Setup State Estimator


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

Next, we should setup our predictor

### Setup Predictor

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

We will also adjust the voltage threshold for the sake of a demo.


```python
batt.parameters['VEOD'] = 3.3
```

With this, we are ready to predict. Let's pull it all together

## Putting it together- Prognostics Example

Now it's time to put it all together.

Typically in a fielded system predictions do not occur every time there is a state estimation. Instead, state estimation happens whenever there's new data, and prediction happens at some lower frequency. 

In some cases the update frequency may be in wall clock time, or after every operational period (e.g., flight). Predictions can also be triggered (or made more frequently) by proximity to event or by the output of a diagnoser. 

In this case we are specifying a certain number of update steps between predictions


```python
PREDICTION_UPDATE_FREQ = 50
```

Next, let's initialize a data structure for storing the results, using the following built-in class:


```python
from progpy.predictors import ToEPredictionProfile
profile = ToEPredictionProfile()
```

Now we'll perform prognostics. We'll loop through the playback data, estimating the state at each time step, and making a prediction at the `PREDICTION_UPDATE_FREQ`.


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

This is an example with playback data. In a real application, the state estimator would be listening to data from a data stream and would be publishing the results to some consumer (e.g., a data bus or directly updating a dispaly)

With our prognostics results, we can now calculate some metrics to analyze the accuracy.

We'll start by calculating the cumulative relative accuracy given the ground truth value. 


```python
cra = profile.cumulative_relative_accuracy(GROUND_TRUTH)
print(f"Cumulative Relative Accuracy for 'EOD': {cra['EOD']}")
```

We'll also generate some plots of the results, given a specific ground truth


```python
GROUND_TRUTH= {'EOD': 1600} 
ALPHA = 0.05
playback_plots = profile.plot(GROUND_TRUTH, ALPHA, True)
```

## Conclusions

**TODO**
