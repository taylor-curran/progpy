# 2. Parameter Estimation

Parameter estimation is used to tune the parameters of a general model so its behavior matches the behavior of a specific system. For example, parameters of the battery model can be tuned to configure the model to describe the behavior of a specific battery.

Generally, parameter estimation is done by tuning the parameters of the model so that simulation (see 1. Simulation) best matches the behavior observed in some available data. In ProgPy, this is done using the progpy.PrognosticsModel.estimate_params() method. This method takes input and output data from one or more runs, and uses scipy.optimize.minimize function to estimate the parameters of the model. For more information, refer to our Documentation [here](https://nasa.github.io/progpy/prog_models_guide.html#parameter-estimation)

A few definitions:
* __`keys`__ `(list[str])`: Parameter keys to optimize
* __`times`__ `(list[float])`: Array of times for each run
* __`inputs`__ `(list[InputContainer])`: Array of input containers where inputs[x] corresponds to times[x]
* __`outputs`__ `(list[OutputContainer])`: Array of output containers where outputs[x] corresponds to times[x]
* __`method`__ `(str, optional)`: Optimization method- see scipy.optimize.minimize for options
* __`tol`__ `(int, optional)`: Tolerance for termination. Depending on the provided minimization method, specifying tolerance sets solver-specific options to tol
* __`error_method`__ `(str, optional)`: Method to use in calculating error. See calc_error for options
* __`bounds`__ `(tuple or dict, optional)`: Bounds for optimization in format ((lower1, upper1), (lower2, upper2), ...) or {key1: (lower1, upper1), key2: (lower2, upper2), ...}
* __`options`__ `(dict, optional)`: Options passed to optimizer. See scipy.optimize.minimize for options

#### Example 1) Simple Example

Now we will show an example demonstrating model parameter estimation. In this example, we estimate the model parameters from data. In general, the data will usually be collected from the physical system or from a different model (model surrogacy). In this case, we will use the example data, below:


```python
times = [0, 1, 2, 3, 4, 5, 6, 7]
inputs = [{}]*8
outputs = [
    {'x': 1.83},
    {'x': 36.5091999066245},
    {'x': 60.05364349596605},
    {'x': 73.23733081022635},
    {'x': 76.47528104941956},
    {'x': 69.9146810161441},
    {'x': 53.74272753819968},
    {'x': 28.39355725512131},
]
```

First, we will import a model from the ProgPy Package. For this example we're using the simple ThrownObject model.


```python
from progpy.models import ThrownObject
```

Now we can build a model with a best guess for the parameters.

We will use a guess that our thrower is 20 meters tall, has a throwing speed of 3.1 m/s, and that acceleration due to gravity is 15 m/s^2. However, given our times, inputs, and outputs, we can clearly tell this is not true! Let's see if parameter estimation can fix this!


```python
m = ThrownObject(thrower_height=20, throwing_speed=3.1, g=15)
```

For this example, we will define specific parameters that we want to estimate.

We can pass the desired parameters to our __keys__ keyword argument.


```python
keys = ['thrower_height', 'throwing_speed', 'g']
```

To really see what `estimate_params()` is doing, we will print out the state before executing the estimation.


```python
# Printing state before
print('Model configuration before')
for key in keys:
    print("-", key, m[key])
print(' Error: ', m.calc_error(times, inputs, outputs, dt=0.1))
```

Notice that the error is quite high. This indicates that the parameters are not accurate.

Now, we will run `estimate_params()` with the data to correct these parameters.


```python
m.estimate_params(times = times, inputs = inputs, outputs = outputs, keys = keys, dt=0.1)
```

Now, let's see what the new parameters are after estimation.


```python
print('\nOptimized configuration')
for key in keys:
    print("-", key, m[key])
print(' Error: ', m.calc_error(times, inputs, outputs, dt=0.1))
```

Sure enough- parameter estimation determined that the thrower's height wasn't 20m, instead was closer to 1.8m, a much more reasonable height!

#### Example 2) Using Tol

An additional feature of the `estimate_params()` function is the tolerance feature, or `tol`. The exact function that the `tol` argument
uses is specific to the method used. For example, the `tol` argument for the `Nelder-Mead` method is the change in the lowest error and its corresponding parameter values between iterations. The difference between iterations for both of these must be below `tol` for parameter estimation to converge.

For example, if in the nth iteration of the optimizer above the best error was __2e-5__ and the cooresponding values were thrower_height=1.8, throwing_speed=40, and g=-9.8 and at the n+1th iteration the best error was __1e-5__ and the cooresponding values were thrower_height=1.85, throwing_speed=39.5, and g=-9.81, then the difference in error would be __1e-5__ and the difference in parameter values would be 

$$\sqrt{(1.85 - 1.8)^2 + (40 - 39.5)^2 + (9 - 9.81)^2} = 0.5025932749$$

In this case, error would meet a tol of __1e-4__, but the parameters would not, so optimization would continue. For more information, see the [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) documentation.

In our previous example, note that our total error was roughly __6e-10__ after the `estimate_params()` call, using the default `tol` of __1e-4__. Now, let us see what happens to the parameters when we pass a tolerance of __1e-6__.


```python
m = ThrownObject(thrower_height=20, throwing_speed=3.1, g=15)
m.estimate_params(times = times, inputs = inputs, outputs = outputs, keys = keys, dt=0.1, tol=1e-6)
print('\nOptimized configuration')
for key in keys:
    print("-", key, m[key])
print(' Error: ', m.calc_error(times, inputs, outputs, dt=0.1))
```

As expected, reducing the tolerance leads to a decrease in the overall error, resulting in more accurate parameters.

Note, if we were to set a high tolerance, such as 10, our error would consequently be very high!

Also note that the tol value is for scipy minimize. It is different but strongly correlated to the result of calc_error. For more information on how the `tol` feature works, please consider scipy's `minimize()` documentation located [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)

You can also adjust the metric that is used to estimate parameters by setting the error_method to a different `calc_error()` method (see example below).
Default is Mean Squared Error (MSE).
See calc_error method for list of options.


```python
m['thrower_height'] = 3.1
m['throwing_speed'] = 29

# Using MAE, or Mean Absolute Error instead of the default Mean Squared Error.
m.estimate_params(times = times, inputs = inputs, outputs = outputs, keys = keys, dt=0.1, tol=1e-9, error_method='MAX_E')
print('\nOptimized configuration')
for key in keys:
    print("-", key, m[key])
print(' Error: ', m.calc_error(times, inputs, outputs, dt=0.1, method='MAX_E'))
```

Note that MAX_E is frequently better at capturing tail behavior in many prognostic models.

#### Example 3) Handling Noise with Multiple Runs

In the previous two examples, we demonstrated how to use `estimate_params()` using a clearly defined ThrownObject model. However, unlike most models, we assumed that there would be no noise!

In this example, we'll show how to use `estimate_params()` with noisy data.

First let's repeat the previous example, this time generating data from a noisy model.


```python
m = ThrownObject(process_noise = 1)
results = m.simulate_to_threshold(save_freq=0.5, dt=('auto', 0.1))

# Resetting parameters to their incorrectly set values.
m['thrower_height'] = 20
m['throwing_speed'] = 3.1
m['g'] = 15
keys = ['thrower_height', 'throwing_speed', 'g']
```


```python
m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, keys = keys)
print('\nOptimized configuration')
for key in keys:
    print("-", key, m[key])
print(' Error: ', m.calc_error(results.times, results.inputs, results.outputs))
```

In this case, the error from calc_error is low, but to have an accurate estimation of the error, we should actually be manually measuring the Absolute Mean Error rather than using calc_error().

The reason being is simple! calc_error() is calculating the error between the simulated and observed data. However, the observed and simulated data in this case are being generated from a model that has noise! In other words, we are comparing the difference of noise to noise, which can lead to inconsistent results!

Let's create a helper function to calculate the Absolute Mean Error between our original and estimated parameters!


```python
# Creating a new model with the original parameters to compare to the model with noise.
true_Values = ThrownObject()

# Function to determine the Absolute Mean Error (AME) of the model parameters.
def AME(m, keys):
    error = 0
    for key in keys:
        error += abs(m[key] - true_Values[key])
    return error
```

Now using our new AME function we see that the error isn't as great as we thought.


```python
AME(m, keys)
```

Note that the error changes every time due to the randomness of noise:


```python
for count in range(10):
    m = ThrownObject(process_noise = 1)
    results = m.simulate_to_threshold(save_freq=0.5, dt=('auto', 0.1))
    
    # Resetting parameters to their originally incorrectly set values.
    m['thrower_height'] = 20
    m['throwing_speed'] = 3.1
    m['g'] = 15

    m.estimate_params(times = results.times, inputs = results.inputs, outputs = results.outputs, keys = keys, dt=0.1)
    error = AME(m, ['thrower_height', 'throwing_speed', 'g'])
    print(f'Estimate Call Number {count} - AME Error {error}')
```

This issue with noise can be overcome with more data. Let's repeat the example above, this time using data from multiple runs. First, let's generate the data:


```python
times, inputs, outputs = [], [], []
m = ThrownObject(process_noise=1)
for count in range(20):
    results = m.simulate_to_threshold(save_freq=0.5, dt=('auto', 0.1))
    times.append(results.times)
    inputs.append(results.inputs)
    outputs.append(results.outputs)
```

Next let's reset the parameters to our incorrect values


```python
m['thrower_height'] = 20
m['throwing_speed'] = 3.1
m['g'] = 15
```

Finally, let's call estimate_params with all the collected data


```python
m.estimate_params(times=times, inputs=inputs, outputs=outputs, keys=keys, dt=0.1)
print('\nOptimized configuration')
for key in keys:
    print("-", key, m[key])
error = AME(m, ['thrower_height', 'throwing_speed', 'g'])
print('AME Error: ', error)
```

Notice that by using data from multiple runs, we are able to produce a lower AME Error than before! This is because we are able to simulate the noise multiple times, which in turn, allows our `estimate_params()` to produce a more accurate result since it is given more data to work with!
