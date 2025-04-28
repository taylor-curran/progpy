# Welcome to ProgPy's Benchmarking Example

The goal of this notebook is to demonstrate benchmarking Prognostic Models. Specifically, we will demonstrate how to benchmark the computational efficiency of model simulation. This is typically what you want to look at when benchmarking models, since simulation is the foundation of state estimation and prediction.

First, we need to import the necessary modules.


```python
from progpy.models import BatteryCircuit
from timeit import timeit
```

The first import is importing a model to benchmark. In this case, ProgPy's BatteryCircuit Model. The second import is of the timeit tool, which will be used to benchmark our model.

Next, let's create our Battery Circuit model.


```python
# Step 1: Create a model object
batt = BatteryCircuit()
```

Then, for our model, we will need to define a future loading function. More information on what a future loading function is and how to use it can be found here: https://nasa.github.io/progpy/prog_models_guide.html#future-loading

Since this is a simple example, we are going to have a constant loading!


```python
# Step 2: Define future loading function 
loading = batt.InputContainer({'i': 2})  # Constant loading
def future_loading(t, x=None):
    # Constant Loading
    return loading
```

Finally, we are ready to benchmark the simulation.

We can do this by using the `timeit()` function and pass in our `simulate_to()` or `simulate_to_threshold()` function for the `stmt` argument. For more information regarding the `timeit()` function, please read its documentation located here: https://docs.python.org/3/library/timeit.html


```python
# Step 3: Benchmark simulation of 600 seconds
def sim():  
    batt.simulate_to(600, future_loading)
time = timeit(sim, number=500)
```

In this example, we are benchmarking the simulation for the BatteryCircuit model up to 600 seconds. Furthermore, we define our `number` argument to be 500 for sake of runtime.

Let's print out the results of the benchmark test!


```python
# Print results
print('Simulation Time: {} ms/sim'.format(time))
```

In this example, we benchmarked the simulation of the BatteryCircuit model up to 600 seconds by utilizing the `time` package!
