# 9. Predicting with Prognostics Models
**A full version of this notebook will be added in release v1.8. In the meatime one section has been included below**

## Predicting a specific event
**A version of this section will be added in release v1.8**

## Prediction horizon

When performing a prediction, it is sometimes desirable to specify a maximum time limit for the prediction, or the prediction `horizon`. This prediction horizon marks the end of the "time of interest" for the prediction. Often this represents the end of a mission or sufficiently far in the future where the user is unconcerned with the events that occur after this time.

The following example illustrates the use of a `horizon` by performing a prediction with uncertainty given a Prognostics Model with a specific prediction horizon. 


We will use the [ThrownObject](https://nasa.github.io/progpy/api_ref/prog_models/IncludedModels.html#thrownobject) model for this example. Once an instance of this class is created, prediction will occur in two steps: 
1) Specifying an initial state 
2) Prediction of future states (with uncertainty) and the times at which the event thresholds will be reached, within the prediction horizon. All events outside the horizon come back as None and are ignored in metrics. 

The results of this prediction will be:
- Predicted future values (inputs, states, outputs, event_states) with uncertainty from prediction
- Time the event is predicted to occur (with uncertainty)

First, let's import the necessary modules.


```python
import numpy as np
from progpy.models.thrown_object import ThrownObject
from progpy.predictors import MonteCarlo
from progpy.uncertain_data import MultivariateNormalDist
from pprint import pprint
```

Next, let's define our model. We'll instantiate a `ThrownObject` model, then we'll initialize the model. 


```python
m = ThrownObject()
initial_state = m.initialize()
```

To predict, we need an initial state. Like in simulation, the initial state defines the starting point from which predictions start. Unlike simulation, prediction uses a distribution of possible states. Here, we define an initial state distribution as a MultiVariateNormalDistribution. 


```python
x = MultivariateNormalDist(initial_state.keys(), initial_state.values(), np.diag([x_i*0.01 for x_i in initial_state.values()]))
```

Next, let's set up a predictor. Here, we'll be using the [MonteCarlo](https://nasa.github.io/progpy/prog_algs_guide.html#prog_algs.predictors.MonteCarlo) Predictor.


```python
mc = MonteCarlo(m)
```

Now, let's perform a prediction. We give the `predict` method the following arguments:
- Distribution of initial samples
- Number of samples for prediction 
- Step size for the prediction 
- Prediction horizon, i.e. time value to predict to



```python
PREDICTION_HORIZON = 7.7
STEP_SIZE = 0.01
NUM_SAMPLES = 500

# Make Prediction
mc_results = mc.predict(x, n_samples=NUM_SAMPLES, dt=STEP_SIZE, horizon = PREDICTION_HORIZON)

```

Let's see the results of the predicted time of event. We'll plot histograms of the distribution of times where `falling` and `impact` occurred. Note that no events occur after 7.7 seconds, since we enforced a prediction horizon at this value. 


```python
metrics = mc_results.time_of_event.metrics()
print("\nPredicted Time of Event:")
pprint(metrics)  # Note this takes some time
fig = mc_results.time_of_event.plot_hist(keys = 'impact')
fig = mc_results.time_of_event.plot_hist(keys = 'falling')
```

Now let's calculate the percentage of each event that occurred before the prediction horizon was met. 


```python
print("\nSamples where falling occurs before horizon: {:.2f}%".format(metrics['falling']['number of samples']/NUM_SAMPLES * 100))
print("\nSamples where impact occurs before horizon: {:.2f}%".format(metrics['impact']['number of samples']/NUM_SAMPLES * 100))
```

All samples reach `falling` before the prediction horizon, but only some of the samples reach `impact`. 

To conclude, in this example, we've shown how to implement a prediction `horizon`. Specifying a prediction horizon defines the time value with which to predict to, and can be used anytime a user is only interested in events that occur before a specific point in time.  
