# 4. Defining new Physics-Based Prognostic Models

All of the past sections describe how to use an existing model. In this section we will describe how to create a new model. This section specifically describes creating a new physics-based model. For training and creating data-driven models see 5. Data-driven Models.

## Linear Models

The easiest model to build is a linear model. Linear models are defined as a linear time series, which can be defined by the following equations:



**The State Equation**:
$$
\frac{dx}{dt} = Ax + Bu + E
$$

**The Output Equation**:
$$
z = Cx + D
$$

**The Event State Equation**:
$$
es = Fx + G
$$

$x$ is `state`, $u$ is `input`, $z$ is `output`, and $es$ is `event state`

Linear Models are defined by creating a new model class that inherits from progpy's LinearModel class and defines the following properties:
* $A$: 2-D np.array[float], dimensions: n_states x n_states. <font color = 'teal'>The state transition matrix. It dictates how the current state affects the change in state dx/dt.</font>
* $B$: 2-D np.array[float], optional (zeros by default), dimensions: n_states x n_inputs. <font color = 'teal'>The input matrix. It dictates how the input affects the change in state dx/dt.</font>
* $C$: 2-D np.array[float], dimensions: n_outputs x n_states. The output matrix. <font color = 'teal'>It determines how the state variables contribute to the output.</font>
* $D$: 1-D np.array[float], optional (zeros by default), dimensions: n_outputs x 1. <font color = 'teal'>A constant term that can represent any biases or offsets in the output.</font>
* $E$: 1-D np.array[float], optional (zeros by default), dimensions: n_states x 1. <font color = 'teal'>A constant term, representing any external effects that are not captured by the state and input.</font>
* $F$: 2-D np.array[float], dimensions: n_es x n_states. <font color = 'teal'>The event state matrix, dictating how state variables contribute to the event state.</font>
* $G$: 1-D np.array[float], optional (zeros by default), dimensions: n_es x 1. <font color = 'teal'>A constant term that can represent any biases or offsets in the event state.</font>
* __inputs__:  list[str] - `input` keys
* __states__:  list[str] - `state` keys
* __outputs__: list[str] - `output` keys
* __events__:  list[str] - `event` keys

We will now utilize our LinearModel to model the classical physics problem throwing an object into the air. This is a common example model, the non-linear version of which (`progpy.examples.ThrownObject`) has been used frequently throughout the examples. This version of ThrownObject will behave nearly identically to the non-linear ThrownObject, except it will not have the non-linear effects of air resistance.

We can create a subclass of LinearModel which will be used to simulate an object thrown, which we will call the ThrownObject Class.

First, some definitions for our Model:

**Events**: (2)
* `falling: The object is falling`
* `impact: The object has hit the ground`

**Inputs/Loading**: (0)
* `None`

**States**: (2)
* `x: Position in space (m)`
* `v: Velocity in space (m/s)`

**Outputs/Measurements**: (1)
* `x: Position in space (m)`

Now, for our keyword arguments:

* <font color = green>__thrower_height : Optional, float__</font>
  * Height of the thrower (m). Default is 1.83 m
* <font color = green>__throwing_speed : Optional, float__</font>
  * Speed at which the ball is thrown (m/s). Default is 40 m/s

With our definitions, we can now create the ThrownObject Model.

First, we need to import the necessary packages.


```python
import numpy as np
from progpy import LinearModel
```

Now we'll define some features of a ThrownObject LinearModel. Recall that all LinearModels follow a set of core equations and require some specific properties (see above). In the next step, we'll define our inputs, states, outputs, and events, along with the $A$, $C$, $E$, and $F$ values.

First, let's consider state transition. For an object thrown into the air without air resistance, velocity would decrease linearly by __-9.81__ 
$\dfrac{m}{s^2}$ due to the effect of gravity, as described below:

 $$\frac{dv}{dt} = -9.81$$

 Position change is defined by velocity (v), as described below:
 
 $$\frac{dx}{dt} = v$$

 Note: For the above equation x is position not state. Combining these equations with the model $\frac{dx}{dt}$ equation defined above yields the A and E matrix defined below. Note that there is no B defined because this model does not have any inputs.


```python
class ThrownObject(LinearModel):
    events = ['impact']
    inputs = []  
    states = ['x', 'v']
    outputs = ['x']
    
    A = np.array([[0, 1], [0, 0]])
    C = np.array([[1, 0]])
    E = np.array([[0], [-9.81]])
    F = None
```

Note that we defined our `A`, `C`, `E`, and `F` values to fit the dimensions that were stated at the beginning of the notebook! Since the parameter `F` is not optional, we have to explicitly set the value as __None__.

Next, we'll define some default parameters for our ThrownObject model.


```python
class ThrownObject(ThrownObject):  # Continue the ThrownObject class
    default_parameters = {
        'thrower_height': 1.83,
        'throwing_speed': 40,
    }
```

In the following cells, we'll define some class functions necessary to perform prognostics on the model.

The `initialize()` function sets the initial system state. Since we have defined the `x` and `v` values for our ThrownObject model to represent position and velocity in space, our initial values would be the thrower_height and throwing_speed parameters, respectively.


```python
class ThrownObject(ThrownObject):
    def initialize(self, u=None, z=None):
        return self.StateContainer({
            'x': self['thrower_height'],
            'v': self['throwing_speed']
            })
```

For our `threshold_met()`, we define the function to return True for event 'falling' when our thrown object model has a velocity value of less than 0 (object is 'falling') and for event 'impact' when our thrown object has a distance from of the ground of less than or equal to 0 (object is on the ground, or has made 'impact').

`threshold_met()` returns a _dict_ of values, if each entry of the _dict_ is __True__, then our threshold has been met!


```python
class ThrownObject(ThrownObject):
    def threshold_met(self, x):
        return {
            'falling': x['v'] < 0,
            'impact': x['x'] <= 0
        }
```

Finally, for our `event_state()`, we will calculate the measurement of progress towards the events. We normalize our values such that they are in the range of 0 to 1, where 0 means the event has occurred.


```python
class ThrownObject(ThrownObject):
    def event_state(self, x): 
        x_max = x['x'] + np.square(x['v'])/(9.81*2)
        return {
            'falling': np.maximum(x['v']/self['throwing_speed'],0),
            'impact': np.maximum(x['x']/x_max,0) if x['v'] < 0 else 1
        }
```

With these functions created, we can now use the `simulate_to_threshold()` function to simulate the movement of the thrown object in air. For more information, see 1. Simulation.


```python
m = ThrownObject()
save = m.simulate_to_threshold(print=True, save_freq=1, events='impact', dt=0.1)
```

__Note__: Because our model takes in no inputs, we have no need to define a future loading function. However, for most models, there would be inputs, and thus a need for a future loading function. For more information on future loading functions and when to use them, please refer to the future loading section in 1. Simulation.

Let's take a look at the outputs of this model


```python
fig = save.outputs.plot(title='generated model')
```

Notice that that plot resembles a parabola, which represents the position of the ball through space as time progresses!

For more information on Linear Models, see the [Linear Model](https://nasa.github.io/progpy/api_ref/prog_models/LinearModel.html) Documentation.

## New State Transition Models

In the previous section, we defined a new prognostic model using the LinearModel class. This can be a powerful tool for defining models that can be described as a linear time series. Physics-based state transition models that cannot be described linearly are constructed by subclassing [progpy.PrognosticsModel](https://nasa.github.io/progpy/api_ref/prog_models/PrognosticModel.html#prog_models.PrognosticsModel). To demonstrate this, we'll create a new model class that inherits from this class. Once constructed in this way, the analysis and simulation tools for PrognosticsModels will work on the new model.

For this example, we'll create a simple state-transition model of an object thrown upward into the air without air resistance. Note that this is the same dynamic system as the linear model example above, but formulated in a different way. 

First, we'll import the necessary packages to create a general prognostics model.


```python
import numpy as np
from progpy import PrognosticsModel
```

Next, we'll define our model class. PrognosticsModels require defining [inputs](https://nasa.github.io/progpy/glossary.html#term-input), [states](https://nasa.github.io/progpy/glossary.html#term-state), [outputs](https://nasa.github.io/progpy/glossary.html#term-output), and [event](https://nasa.github.io/progpy/glossary.html#term-event) keys. As in the above example, the states include position (`x`) and velocity(`v`) of the object, the output is position (`x`), and the events are `falling` and `impact`. 

Note that we define this class as `ThrownObject_ST` to distinguish it as a state-transition model compared to the previous linear model class. 


```python
class ThrownObject_ST(PrognosticsModel):
    """
    Model that simulates an object thrown into the air without air resistance
    """

    inputs = [] # no inputs, no way to control
    states = [
        'x', # Position (m) 
        'v'  # Velocity (m/s)
        ]
    outputs = [ # Anything we can measure
        'x' # Position (m)
    ]
    events = [
        'falling', # Event- object is falling
        'impact' # Event- object has impacted ground
    ]
```

Next, we'll add some default parameter definitions. These values can be overwritten by passing parameters into the constructor. 


```python
class ThrownObject_ST(ThrownObject_ST):

    default_parameters = {
        'thrower_height': 1.83, # default height 
        'throwing_speed': 40, # default speed
        'g': -9.81,  # Acceleration due to gravity (m/s^2)
    }
```

All prognostics models require some specific class functions. We'll define those next. 

First, we'll need to add the functionality to set the initial state of the system. There are two ways to provide the logic to initialize model state. 

1. Provide the initial state in `parameters['x0']`, or 
2. Provide an `initialize` function 

The first method here is preferred. If `parameters['x0']` are defined, there is no need to explicitly define an initialize method, and these parameter values will be used as the initial state. 

However, there are some cases where the initial state is a function of the input (`u`) or output (`z`) (e.g. a use-case where the input is also a state). In this case, an explicitly defined `initialize` method is required. 

Here, we'll set our initial state by defining an `initialize` function. In the code below, note that the function can take arguments for both input `u` and output `z`, though these are optional. 

Note that for this example, defining initialize in this way is not necessary. We could have simply defined `parameters['x0']`. However, we choose to use this method for ease when using the `derived_params` feature, discussed in the next section. 


```python
class ThrownObject_ST(ThrownObject_ST):

    def initialize(self, u=None, z=None):
        return self.StateContainer({
            'x': self['thrower_height'],  # Thrown, so initial altitude is height of thrower
            'v': self['throwing_speed']   # Velocity at which the ball is thrown - this guy is a professional baseball pitcher
            })
```

Next, the PrognosticsModel class requires that we define how the state transitions throughout time. For continuous models, this is defined with the method `dx`, which calculates the first derivative of the state at a specific time. For discrete systems, this is defined with the method `next_state`, using the state transition equation for the system. When possible, it is recommended to use the continuous (`dx`) form, as some algorithms will only work on continuous models.

Here, we use the equations for the derivatives of our system (i.e., the continuous form).


```python
class ThrownObject_ST(ThrownObject_ST):

    def dx(self, x, u):
        return self.StateContainer({
                'x': x['v'], 
                'v': self['g']})  # Acceleration of gravity
```

Next, we'll define the `output` method, which will calculate the output (i.e., measurable values) given the current state. Here, our output is position (`x`). 


```python
class ThrownObject_ST(ThrownObject_ST):
     
    def output(self, x):
        return self.OutputContainer({'x': x['x']})
```

The next method we define is [`event_state`](https://nasa.github.io/progpy/glossary.html#term-event-state). As before, 
`event_state` calculates the progress towards the events. Normalized to be between 0 and 1, 1 means there is no progress towards the event and 0 means the event has occurred. 


```python
class ThrownObject_ST(ThrownObject_ST):
    
    def event_state(self, x): 
        # Use speed and position to estimate maximum height
        x_max = x['x'] + np.square(x['v'])/(-self['g']*2)
        # 1 until falling begins
        x_max = np.where(x['v'] > 0, x['x'], x_max)
        return {
            'falling': max(x['v']/self['throwing_speed'],0),  # Throwing speed is max speed
            'impact': max(x['x']/x_max,0)  # 1 until falling begins, then it's fraction of height
        }
```

At this point, we have defined all necessary information for the PrognosticsModel to be complete. There are other methods that can additionally be defined (see the [PrognosticsModel](https://nasa.github.io/progpy/api_ref/prog_models/PrognosticModel.html) documentation for more information) to provide further configuration for new models. We'll highlight some of these in the following sections. 

As an example of one of these, we additionally define a `threshold_met` equation. Note that this is optional. Leaving `threshold_met` empty will use the event state to define thresholds (threshold = event state == 0). However, this implementation is more efficient, so we include it. 

Here, we define `threshold_met` in the same way as our linear model example. `threshold_met` will return a _dict_ of values, one for each event. Threshold is met when all dictionary entries are __True__. 


```python
class ThrownObject_ST(ThrownObject_ST):

    def threshold_met(self, x):
        return {
            'falling': x['v'] < 0, # Falling occurs when velocity becomes negative
            'impact': x['x'] <= 0 # Impact occurs when the object hits the ground, i.e. position is <= 0
        }
```

With that, we have created a new ThrownObject state-transition model. 

Now let's can test our model through simulation. First, we'll create an instance of the model.


```python
m_st = ThrownObject_ST()
```

We'll start by simulating to impact. We'll specify the `events` to specifically indicate we are interested in impact. For more information on simulation, see 1. Simulation. 


```python
# Simulate to impact
event = 'impact'
simulated_results = m_st.simulate_to_threshold(events=event, dt=0.005, save_freq=1, print = True)

# Print result: 
print('The object hit the ground in {} seconds'.format(round(simulated_results.times[-1],2)))
```

To summarize this section, we have illustrated how to construct new physics-based models by subclassing from [progpy.PrognosticsModel](https://nasa.github.io/progpy/api_ref/prog_models/PrognosticModel.html#prog_models.PrognosticsModel). Some elements (e.g. inputs, states, outputs, events keys; methods for initialization, dx/next_state, output, and event_state) are required. Models can be additionally configured with additional methods and parameters.

Note that in this example, we defined each part one piece at a time, recursively subclassing the partially defined class. This was done to illustrate the parts of the model. In reality, all methods and properties would be defined together in a single class definition. 

## Derived Parameters

In the previous section, we constructed a new model from scratch by subclassing from [progpy.PrognosticsModel](https://nasa.github.io/progpy/api_ref/prog_models/PrognosticModel.html#prog_models.PrognosticsModel) and specifying all of the necessary model components. An additional optional feature of PrognosticsModels is derived parameters, illustrated below. 

A derived parameter is a parameter (see parameter section in 1. Simulation) that is a function of another parameter. For example, in the case of a thrown object, one could assume that throwing speed is a function of thrower height, with taller throwing height resulting in faster throwing speeds. In the electrochemistry battery model (see 3. Included Models), there are parameters for the maximum and minimum charge at the surface and bulk, and these are dependent on the capacity of the battery (i.e. another parameter, qMax). When such derived parameters exist, they must be updated whenever the parameters they depend on are updated. In PrognosticsModels, this is achieved with the `derived_params` feature. 

This feature can also be used to cache combinations of parameters that are used frequently in state transition or other model methods. Creating lumped parameters using `derived_params` causes them to be calculated once when configuring, instead of each time step in simulation or prediction. 

For this example, we will use the `ThrownObject_ST` model created in the previous section. We will extend this model to include a derived parameter, namely `throwing_speed` will be dependent on `thrower_height`.

To implement this, we must first define a function for the relationship between the two parameters. We'll assume that `throwing_speed` is a linear function of `thrower_height`. 


```python
def update_thrown_speed(params):
    return {
        'throwing_speed': params['thrower_height'] * 21.85
    }  
    # Note: one or more parameters can be changed in these functions, whatever parameters are changed are returned in the dictionary
```

Next, we'll define the parameter callbacks, so that `throwing_speed` is updated appropriately any time that `thrower_height` changes. The following effectively tells the derived callbacks feature to call the `update_thrown_speed` function whenever the `thrower_height` changes. 


```python
class ThrownObject_ST(ThrownObject_ST):

    param_callbacks = {
        'thrower_height': [update_thrown_speed]
    }
```

You can also have more than one function be called when a single parameter is changed. You would do this by adding the additional callbacks to the list (e.g., 'thrower_height': [update_thrown_speed, other_fcn])

We have now added the capability for `throwing_speed` to be a derived parameter. Let's try it out. First, we'll create an instance of our class and print out the default parameters. 


```python
obj = ThrownObject_ST()
print("Default Settings:\n\tthrower_height: {}\n\tthrowing_speed: {}".format(obj['thrower_height'], obj['throwing_speed']))
```

Now, let's change the thrower height. If our derived parameters work correctly, the thrower speed should change accordingly. 


```python
obj['thrower_height'] = 1.75  # Our thrower is 1.75 m tall
print("\nUpdated Settings:\n\tthrower_height: {}\n\tthowing_speed: {}".format(obj['thrower_height'], obj['throwing_speed']))
```

As we can see, when the thrower height was changed, the throwing speed was re-calculated too. 

In this example, we illustrated how to use the `derived_params` feature, which allows a parameter to be a function of another parameter. 

## Direct Models

In the previous sections, we illustrated how to create and use state-transition models, or models that use state transition differential equations to propagate the state forward. In this example, we'll explore another type of model implemented within ProgPy - Direct Models. 

Direct models estimate the time of event directly from the system state and future load, rather than through state transitions. This approach is particularly useful for physics-based models where the differential equations of state transitions can be explicitly solved, or for data-driven models that map sensor data directly to the time of an event. When applicable, using a direct model approach provides a more efficient way to estimate the time of an event, especially for events that occur later in the simulation. 

To illustrate this concept, we will extend the state-transition model, `ThrownObject_ST`, defined above, to create a new model class, `DirectThrownObject`. The dynamics of a thrown object lend easily to a direct model, since we can solve the differential equations explicitly to estimate the time at which the events occur. 

Recall that our physical system is described by the following differential equations: 
\begin{align*}
\frac{dx}{dt} &= v \\ \\
\frac{dv}{dt} &= -g 
\end{align*}

which can be solved explicity, given initial position $x_0$ and initial velocity $v_0$, to get:
\begin{align*}
x(t) &= -\frac{1}{2} gt^2 + v_0 t + x_0 \\ \\ 
v(t) &= -gt + v_0
\end{align*}

Setting these equations to 0 and solving for time, we get the time at which the object hits the ground and begins falling, respectively. 

To construct our direct model, we'll extend the `ThrownObject_ST` model to additionally include the method [time_to_event](https://nasa.github.io/progpy/api_ref/prog_models/PrognosticModel.html#prog_models.PrognosticsModel.time_of_event). This method will calculate the time at which each event occurs (i.e., time when the event threshold is met), based on the equations above. `time_of_event` must be implemented by any direct model. 


```python
class DirectThrownObject(ThrownObject_ST):
    
    def time_of_event(self, x, *args, **kwargs):
        # calculate time when object hits ground given x['x'] and x['v']
        # 0 = x0 + v0*t - 0.5*g*t^2
        g = self['g']
        t_impact = -(x['v'] + np.sqrt(x['v']*x['v'] - 2*g*x['x']))/g

        # 0 = v0 - g*t
        t_falling = -x['v']/g
                
        return {'falling': t_falling, 'impact': t_impact}

```

With this, our direct model is created. Note that adding `*args` and `**kwargs` is optional. Having these arguments makes the function interchangeable with other models which may have arguments or keyword arguments. 

Now let's test out this capability. To do so, we'll use the `time` package to compare the direct model to our original timeseries model. 

Let's start by creating an instance of our timeseries model, calculating the time of event, and timing this computation. Note that for a state transition model, `time_of_event` still returns the time at which `threshold_met` returns true for each event, but this is calculated by simulating to threshold.


```python
import time 

m_timeseries = ThrownObject_ST()
x = m_timeseries.initialize()
print(m_timeseries.__class__.__name__, "(Direct Model)" if m_timeseries.is_direct else "(Timeseries Model)")
tic = time.perf_counter()
print('Time of event: ', m_timeseries.time_of_event(x, dt = 0.05))
toc = time.perf_counter()
print(f'execution: {(toc-tic)*1000:0.4f} milliseconds')
```

Now let's do the same using our direct model implementation. In this case, when `time_to_event` is called, the event time will be estimated directly from the state, instead of through simulation to threshold. 

Note that a limitation of a direct model is that you cannot get intermediate states (i.e., save_pts or save_freq) since the time of event is calculated directly. 


```python
m_direct = DirectThrownObject()
x = m_direct.initialize()  # Using Initial state
# Now instead of simulating to threshold, we can estimate it directly from the state, like so
print('\n', m_direct.__class__.__name__, "(Direct Model)" if m_direct.is_direct else "(Timeseries Model)")
tic = time.perf_counter()
print('Time of event: ', m_direct.time_of_event(x))
toc = time.perf_counter()
print(f'execution: {(toc-tic)*1000:0.4f} milliseconds')
```

Notice that execution is significantly faster for the direct model. Furthermore, the result is actually more accurate, since it's not limited by the timestep (see dt section in 1. Simulation). These observations will be even more pronounced for events that occur later in the simulation. 

It's important to note that this is a very simple example, as there are no inputs. For models with inputs, future loading must be provided to `time_of_event` (see the Future Loading section in 1. Simulation). In these cases, most direct models will encode or discretize the future loading profile to use it in a direct estimation of time of event.

In the example provided, we have illustrated how to use a direct model. Direct models are a powerful tool for estimating the time of an event directly from the system state. By avoiding the process of state transitions, direct models can provide more efficient event time estimates. Additionally, the direct model approach is not limited to physics-based models. It can also be applied to data-driven models that can map sensor data directly to the time of an event. 

In conclusion, direct models offer an efficient and versatile approach for prognostics modeling, enabling faster and more direct estimations of event times. 

## Matrix Data Access Feature

In the above models, we have used dictionaries to represent the states. For example, in the implementation of `ThrownObject_ST` above, see how `dx` is defined with a StateContainer dictionary. While all models can be constructed using dictionaries in this way, some dynamical systems allow for the state of the system to be represented with a matrix. For such use-cases, ProgPy has an advanced *matrix data access feature* that provides a more efficient way to define these models.

In ProgPy's implementation, the provided model.StateContainer, InputContainer, and OutputContainers can be treated as dictionaries but use an underlying matrix. This is important for some applications like surrogate and machine-learned models where the state is represented by a tensor. ProgPy's *matrix data access feature* allows the matrices to be used directly. Simulation functions propagate the state using the matrix form, preventing the inefficiency of having to convert to and from dictionaries. Additionally, this implementation is faster than recreating the StateContainer each time, especially when updating inplace.

In this example, we'll illustrate how to use the matrix data access feature. We'll continue with our ThrownObject system, and create a model to simulate this using matrix notation (instead of dictionary notation as in the standard model, seen above in `ThrownObject_ST`). The implementation of the model is comparable to a standard model, except that it uses matrix operations within each function, as seen below. 

First, the necessary imports.


```python
import numpy as np
from progpy import PrognosticsModel
```

To use the matrix data access feature, we'll subclass from our state-transition model defined above, `ThrownObject_ST`. Our new model will therefore inherit the default parameters and methods for initialization, output, threshold met, and event state. 

To use the matrix data access feature, we'll use matrices to define how the state transitions. Since we are working with a discrete version of the system now, we'll define the `next_state` method, and this will override the `dx` method in the parent class. 

In the following, we will use the matrix version for each variable, accessed with `.matrix`. We implement this within `next_state`, but this feature can also be used in other functions. Here, both `x.matrix` and `u.matrix` are column vectors, and `u.matrix` is in the same order as model.inputs.


```python
class ThrownObject_MM(ThrownObject_ST):

    def next_state(self, x, u, dt):

        A = np.array([[0, 1], [0, 0]])  # State transition matrix
        B = np.array([[0], [self['g']]])  # Acceleration due to gravity
        x.matrix += (np.matmul(A, x.matrix) + B) * dt

        return x
```

Our model is now specified. Let's try simulating with it.

First, we'll create an instance of the model.


```python
m_matrix = ThrownObject_MM()
```

Now, let's simulate to threshold. We'll also time the simulation so we can compare with the non-matrix state-transition model below. 


```python
import time 

tic_matrix = time.perf_counter()
# Simulate to threshold 
m_matrix.simulate_to_threshold(
        print = True, 
        events = 'impact', 
        dt = 0.1, 
        save_freq = 1)
toc_matrix = time.perf_counter()
```

Our matrix notation was successful in simulating the thrown object's behavior throughout time. 

Finally, let's simulate the non-matrix version to compare computation time. 




```python
tic_st = time.perf_counter()
m_st.simulate_to_threshold(
        print = True, 
        events = 'impact', 
        dt = 0.1, 
        save_freq = 1)
toc_st = time.perf_counter()

print(f'Matrix execution: {(toc_matrix-tic_matrix)*1000:0.4f} milliseconds')
print(f'Non-matrix execution: {(toc_st-tic_st)*1000:0.4f} milliseconds')
```

As we can see, for this system, using the matrix data access feature is computationally faster than a standard state-transition matrix that uses dictionaries. 

As illustrated here, the matrix data access feature is an advanced capability that represents the state of a system using matrices. This can provide efficiency for use-cases where the state is easily represented by a tensor and operations are defined by matrices.

## State Limits

In real-world physical systems, there are often constraints on what values the states can take. For example, in the case of a thrown object, if we define our reference frame with the ground at a position of $x=0$, then the position of the object should only be greater than or equal to 0, and should never take on negative values. In ProgPy, we can enforce constraints on the range of each state for a state-transition model using the [state limits](https://nasa.github.io/progpy/prog_models_guide.html#state-limits)  attribute. 

To illustrate the use of `state_limits`, we'll use our thrown object model `ThrownObject_ST`, created in an above section. 


```python
m_limits = ThrownObject_ST()
```

Before adding state limits, let's take a look at the standard model without state limits. We'll consider the event of `impact`, and simulate the object to threshold.


```python
event = 'impact'
simulated_results = m_limits.simulate_to_threshold(events=event, dt=0.005, save_freq=1)

print('Example: No State Limits')
for i, state in enumerate(simulated_results.states):
    print(f'State {i}: {state}')
print()
```

Notice that at the end of the simulation, the object's position (`x`) is negative. This doesn't make sense physically, since the object cannot fall below ground level (at $x=0$).

To avoid this, and keep the state in a realistic range, we can change the `state_limits` attribute of the model. The `state_limits` attribute is a dictionary that contains the state limits for each state. The keys of the dictionary are the state names, and the values are tuples that contain the lower and upper limits of the state. 

In our Thrown Object model, our states are position, which can range from 0 to infinity, and velocity, which we'll limit to not exceed the speed of light.


```python
# Import inf
from math import inf

m_limits.state_limits = {
    # object position may not go below ground height
    'x': (0, inf),

    # object velocity may not exceed the speed of light
    'v': (-299792458, 299792458)
}
```

Now that we've specified the ranges for our state values, let's try simulating again. 


```python
event = 'impact'
simulated_results = m_limits.simulate_to_threshold(events=event, dt=0.005, save_freq=1)

print('Example: With State Limits')
for i, state in enumerate(simulated_results.states):
    print(f'State {i}: {state}')
print()
```

Notice that now the position (`x`) becomes 0 but never reaches a negative value. This is because we have defined a state limit for the `x` state that prevents it from going below 0. Also note that a warning is provided to notify the user that a state value was limited. 

Let's try a more complicated example. This time, we'll try setting the initial position value to be a number outside of its bounds. 


```python
x0 = m_limits.initialize(u = {}, z = {})
x0['x'] = -1 # Initial position value set to an unrealistic value of -1

simulated_results = m_limits.simulate_to_threshold(events=event, dt=0.005, save_freq=1, x = x0)

# Print states
print('Example 2: With -1 as initial x value')
for i, state in enumerate(simulated_results.states):
    print('State ', i, ': ', state)
print()
```

Notice that the simulation stops after just two iterations. In this case, the initial position value is outside the state limit. On the first iteration, the position value is therefore adjusted to be within the appropriate range of 0 to $\infty$. Since we are simulating to impact, which is defined as when position is 0, the threshold is immediately satisfied and the simulation stops. 

Finally, note that limits can also be applied manually using the `apply_limits` function. 


```python
x = {'x': -5, 'v': 3e8}  # Too fast and below the ground
print('\t Pre-limit: {}'.format(x))

x = m_limits.apply_limits(x)
print('\t Post-limit: {}'.format(x))
```

In conclusion, setting appropriate [state limits](https://nasa.github.io/progpy/prog_models_guide.html#state-limits)  is crucial in creating realistic and accurate state-transition models. It ensures that the model's behavior stays within the constraints of the physical system. The limits should be set based on the physical or practical constraints of the system being modeled. 

As a final note, state limits are especially important for state estimation (to be discussed in the State Estimation section), as it will force the state estimator to only consider states that are possible or feasible. State estimation will be described in more detail in section 08. State Estimation. 

## Custom Events

In the examples above, we have focused on the simple event of a thrown object hitting the ground or reaching `impact`. In this section, we highlight additional uses of ProgPy's generalizable concept of `events`. 

The term [events](https://nasa.github.io/progpy/prog_models_guide.html#events) is used to describe something to be predicted. Generally in the PHM community, these are referred to as End of Life (EOL). However, they can be much more. 

In ProgPy, events can be anything that needs to be predicted. Systems will often have multiple failure modes, and each of these modes can be represented by a separate event. Additionally, events can also be used to predict other events of interest other than failure, such as special system states or warning thresholds. Thus, `events` in ProgPy can represent End of Life (EOL), End of Mission (EOM), warning thresholds, or any Event of Interest (EOI). 

There are a few components of the model that must be specified in order to define events:

1. The `events` property defines the expected events 

2. The `threshold_met` method defines the conditions under which an event occurs 

3. The `event_state` method returns an estimate of progress towards the threshold 

Note that because of the interconnected relationship between `threshold_met` and `event_state`, it is only required to define one of these. However, it is generally beneficial to specify both. 

To illustrate this concept, we will use the `BatteryElectroChemEOD` model (see section 03. Included Models). In the standard implementation of this model, the defined event is `EOD` or End of Discharge. This occurs when the voltage drops below a pre-defined threshold value. The State-of-Charge (SOC) of the battery is the event state for the EOD event. Recall that event states (and therefore SOC) vary between 0 and 1, where 1 is healthy and 0 signifies the event has occurred. 

Suppose we have the requirement that our battery must not fall below 5% State-of-Charge. This would correspond to an `EOD` event state of 0.05. Additionally, let's add events for two warning thresholds, a $\text{\textcolor{yellow}{yellow}}$ threshold at 15% SOC and a $\text{\textcolor{red}{red}}$ threshold at 10% SOC. 

To define the model, we'll start with the necessary imports.


```python
import matplotlib.pyplot as plt
from progpy.loading import Piecewise
from progpy.models import BatteryElectroChemEOD
```

Next, let's define our threshold values. 


```python
YELLOW_THRESH = 0.15 # 15% SOC
RED_THRESH = 0.1 # 10% SOC
THRESHOLD = 0.05 # 5% SOC
```

Now we'll create our model by subclassing from the `BatteryElectroChemEOD` model. First, we'll re-define `events` to include three new events for our two warnings and new threshold value, as well as the event `EOD` from the parent class.


```python
class BattNewEvent(BatteryElectroChemEOD):
    events = BatteryElectroChemEOD.events + ['EOD_warn_yellow', 'EOD_warn_red', 'EOD_requirement_threshold']

```

Next, we'll override the `event_state` method to additionally include calculations for progress towards each of our new events. We'll add yellow, red, and failure states by scaling the EOD state. We scale so that the threshold SOC is 0 at their associated events, while SOC of 1 is still 1. For example, for yellow, we want `EOD_warn_yellow` to be 1 when SOC is 1, and 0 when SOC is 0.15 or lower. 


```python
class BattNewEvent(BattNewEvent):
    
    def event_state(self, state):
        # Get event state from parent
        event_state = super().event_state(state)

        # Add yellow, red, and failure states by scaling EOD state
        event_state['EOD_warn_yellow'] = (event_state['EOD']-YELLOW_THRESH)/(1-YELLOW_THRESH) 
        event_state['EOD_warn_red'] = (event_state['EOD']-RED_THRESH)/(1-RED_THRESH)
        event_state['EOD_requirement_threshold'] = (event_state['EOD']-THRESHOLD)/(1-THRESHOLD)

        # Return
        return event_state
```

Finally, we'll override the `threshold_met` method to define when each event occurs. Based on the scaling in `event_state` each event is reached when the corresponding `event_state` value is less than or equal to 0. 


```python
class BattNewEvent(BattNewEvent):
    def threshold_met(self, x):
        # Get threshold met from parent
        t_met = super().threshold_met(x)

        # Add yell and red states from event_state
        event_state = self.event_state(x)
        t_met['EOD_warn_yellow'] = event_state['EOD_warn_yellow'] <= 0
        t_met['EOD_warn_red'] = event_state['EOD_warn_red'] <= 0
        t_met['EOD_requirement_threshold'] = event_state['EOD_requirement_threshold'] <= 0

        return t_met
```

With this, we have defined the three key model components for defining new events. 

Let's test out the model. First, create an instance of it. 


```python
m = BattNewEvent()
```

Recall that the battery model takes input of current. We will use a piecewise loading scheme (see 01. Simulation).


```python
# Variable (piecewise) future loading scheme
future_loading = Piecewise(
        m.InputContainer,
        [600, 900, 1800, 3000],
        {'i': [2, 1, 4, 2, 3]})
```

Now we can simulate to threshold and plot the results.  


```python
simulated_results = m.simulate_to_threshold(future_loading, events='EOD', print = True)

simulated_results.event_states.plot()
plt.show()
```

Here, we can see the SOC plotted for the different events throughout time. The yellow warning (15% SOC) reaches threshold first, followed by the red warning (10% SOC), new EOD threshold (5% SOC), and finally the original EOD value. 

In this section, we have illustrated how to define custom [events](https://nasa.github.io/progpy/prog_models_guide.html#events) for prognostics models. Events can be used to define anything that a user is interested in predicting, including common values like Remaining Useful Life (RUL) and End of Discharge (EOD), as well as other values like special intermediate states or warning thresholds. 

## Serialization 

ProgPy includes a feature to serialize models, which we highlight in this section. 

Model serialization has a variety of purposes. For example, serialization allows us to save a specific model or model configuration to a file to be loaded later, or can aid us in sending a model to another machine over a network connection. Some users maintain a directory or repository of configured models representing specific systems in their stock.

In this section, we'll show how to serialize and deserialize model objects using `pickle` and `JSON` methods. 

First, we'll import the necessary modules.


```python
import matplotlib.pyplot as plt
import pickle
import numpy as np
from progpy.models import BatteryElectroChemEOD
from progpy.loading import Piecewise
```

For this example, we'll use the BatteryElectroChemEOD model. We'll start by creating a model object. 


```python
batt = BatteryElectroChemEOD()
```

First, we'll serialize the model in two different ways using 1) `pickle` and 2) `JSON`. Then, we'll plot the results from simulating the deserialized models to show equivalence of the methods. 

To save using the `pickle` package, we'll serialize the model using the `dump` method. Once saved, we can then deserialize using the `load` method. In practice, deserializing will likely occur in a different file or in a later use-case, but here we deserialize to show equivalence of the saved model. 


```python
pickle.dump(batt, open('save_pkl.pkl', 'wb')) # Serialize model
load_pkl = pickle.load(open('save_pkl.pkl', 'rb')) # Deserialize model 
```

Next, we'll serialize using the `to_json` method. We deserialize by calling the model directly with the serialized result using the `from_json` method.


```python
save_json = batt.to_json() # Serialize model
json_1 = BatteryElectroChemEOD.from_json(save_json) # Deserialize model
```

Note that the serialized result can also be saved to a text file and uploaded for later use. We demonstrate this below:


```python
txtFile = open("save_json.txt", "w")
txtFile.write(save_json)
txtFile.close()

with open('save_json.txt') as infile: 
    load_json = infile.read()

json_2 = BatteryElectroChemEOD.from_json(load_json)
```

We have now serialized and deserialized the model using `pickle` and `JSON` methods. Let's compare the resulting models. To do so, we'll use ProgPy's [simulation](https://nasa.github.io/progpy/prog_models_guide.html#simulation) to simulate the model to threshold and compare the results. 

First, we'll need to define our [future loading profile](https://nasa.github.io/progpy/prog_models_guide.html#future-loading) using the PiecewiseLoad class. 


```python
# Variable (piecewise) future loading scheme
future_loading = Piecewise(
        batt.InputContainer,
        [600, 1000, 1500, 3000],
        {'i': [3, 2, 1.5, 4]})
```

Now, let's simulate each model to threshold using the `simulate_to_threshold` method. 


```python
# Original model 
results_orig = batt.simulate_to_threshold(future_loading, save_freq = 1)
# Pickled version  
results_pkl = load_pkl.simulate_to_threshold(future_loading, save_freq = 1)
# JSON versions
results_json_1 = json_1.simulate_to_threshold(future_loading, save_freq = 1)
results_json_2 = json_2.simulate_to_threshold(future_loading, save_freq = 1)

```

Finally, let's plot the results for comparison.


```python
voltage_orig = [results_orig.outputs[iter]['v'] for iter in range(len(results_orig.times))]
voltage_pkl = [results_pkl.outputs[iter]['v'] for iter in range(len(results_pkl.times))]
voltage_json_1 = [results_json_1.outputs[iter]['v'] for iter in range(len(results_json_1.times))]
voltage_json_2 = [results_json_2.outputs[iter]['v'] for iter in range(len(results_json_2.times))]

plt.plot(results_orig.times,voltage_orig,'-b',label='Original surrogate') 
plt.plot(results_pkl.times,voltage_pkl,'--r',label='Pickled serialized surrogate') 
plt.plot(results_json_1.times,voltage_json_1,'-.g',label='First JSON serialized surrogate') 
plt.plot(results_json_2.times, voltage_json_2, '--y', label='Second JSON serialized surrogate')
plt.legend()
plt.xlabel('Time (sec)')
plt.ylabel('Voltage (volts)')
```

All of the voltage curves overlap, showing that the different serialization methods produce the same results. 

Additionally, we can compare the output arrays directly, to ensure equivalence. 


```python
import numpy as np

# Check if the arrays are the same
are_arrays_same = np.array_equal(voltage_orig, voltage_pkl) and \
                  np.array_equal(voltage_orig, voltage_json_1) and \
                  np.array_equal(voltage_orig, voltage_json_2)

print(f"The simulated results from the original and serialized models are {'identical. This means that our serialization works!' if are_arrays_same else 'not identical. This means that our serialization does not work.'}")
```

To conclude, we have shown how to serialize models in ProgPy using both `pickle` and `JSON` methods. Understanding how to serialize and deserialize models can be a powerful tool for prognostics developers. It enables the saving of models to a disk and the re-loading of these models back into memory at a later time. 

## Example - Simplified Battery Model

This is an example of a somewhat more complicated model, in this case a battery. We will be implementing the simplified battery model introduced by Gina Sierra, et. al. (https://www.sciencedirect.com/science/article/pii/S0951832018301406).

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

Note that this model can be extended by changing the parameters ecrit and r to steady states. This will help the model account for the effects of aging, since they will be estimated with each state estimation step.

## Conclusions

In these examples, we have described how to create new physics-based models. We have illustrated how to construct a generic physics-based model, as well as highlighted some specific types of models including linear models and direct models. We highlighted the matrix data access feature for using matrix operations more efficiently. Additionally, we discussed a few important components of any prognostics model including derived parameters, state limits, and events. 

With these tools, users are well-equipped to build their own prognostics models for their specific physics-based use-cases. In the next example, we'll discuss how to create data-driven models.
