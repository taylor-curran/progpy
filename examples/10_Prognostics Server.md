# Prognostics Server (prog_server)

The ProgPy Server (prog_server) is a simplified implementation of a Service-Oriented Architecture (SOA) for performing prognostics (estimation of time until events and future system states) of engineering systems. prog_server is a wrapper around the ProgPy package, allowing one or more users to access the features of these packages through a REST API. The package is intended to be used as a research tool to prototype and benchmark Prognostics As-A-Service (PaaS) architectures and work on the challenges facing such architectures, including Generality, Communication, Security, Environmental Complexity, Utility, and Trust.

The ProgPy Server is actually two packages, prog_server and prog_client. The prog_server package is a prognostics server that provides the REST API. The prog_client package is a python client that provides functions to interact with the server via the REST API.

**TODO(CT): IMAGE- server with clients**

## Installing

prog_server can be installed using pip

```console
$ pip install prog_server
```

## Starting prog_server

prog_server can be started 2 ways. Through command line or programatically (i.e., in a python script). Once the server is started it will take a short time to initialize, then it will start receiving requests for sessions from clients using prog_client, or interacting directly using the REST interface.

### Starting prog_server in command line
Generally, you can start the prog_server by running the module, like this:

```console
$ python -m prog_server
```

Note that you can force the server to start in debug mode using the `debug` flag. For example, `python -m prog_server --debug`

### Starting prog_server programatically
There are two methods to start the prog_server in python. The first, below, is non-blocking allowing users to perform other functions while the server is running.


```python
import prog_server
prog_server.start()
```

When starting a server, users can also provide arguments to customize the way the server runs. Here are the main arguments used by 

* host (str): Server host address. Defaults to ‘127.0.0.1’
* port (int): Server port address. Defaults to 8555
* debug (bool): If the server is to be started in debug mode

Now prog_server is ready to start receiving session requests from users. The server can also be stopped using the stop() function


```python
prog_server.stop()
```

prog_server can also be started in blocked mode using the following command:

```python
>>> prog_server.run()
```

We will not execute it here, because it would block execution in this notebook until force quit.

For details on all supported arguments see [API Doc](https://nasa.github.io/progpy/api_ref/prog_server/prog_server.html#prog_server.start)

The basis of prog_server is the session. Each user creates one or more session. These sessions are each a request for prognostic services. Then the user can interact with the open session. You'll see examples of this in the future sections.

Let's restart the server (so it can be used with the below examples)


```python
prog_server.start()
```

## Using prog_server with prog_client
For users using python, prog_server can be interacted with using the prog_client package distributed with progpy. This section describes a few examples using prog_client and prog_server together.

Before using prog_client import the package:


```python
import prog_client
```

### Example: Online Prognostics
This example creates a session with the server to run prognostics for a Thrown Object, a simplified model of an object thrown into the air. Data is then sent to the server and a prediction is requested. The prediction is then displayed.

**Note: before running this example, make sure prog_server is running**

The first step is to open a session with the server. This starts a session for prognostics with the ThrownObject model, with default parameters. The prediction configuration is updated to have a save frequency of every 1 second.


```python
session = prog_client.Session('ThrownObject', pred_cfg={'save_freq': 1})
print(session)  # Printing the Session Information
```

If you were to re-run the lines above, it would start a new session, with a new number.

Next, we need to prepare the data we will use for this example. The data is a dictionary, and the keys are the names of the inputs and outputs in the model with format (time, value).

Note: in an actual application, the data would be received from a sensor or other source. The structure below is used to emulate the sensor.


```python
example_data = [
    (0, {'x': 1.83}), 
    (0.1, {'x': 5.81}), 
    (0.2, {'x': 9.75}), 
    (0.3, {'x': 13.51}), 
    (0.4, {'x': 17.20}), 
    (0.5, {'x': 20.87}), 
    (0.6, {'x': 24.37}), 
    (0.7, {'x': 27.75}), 
    (0.8, {'x': 31.09}), 
    (0.9, {'x': 34.30}), 
    (1.0, {'x': 37.42}),
    (1.1, {'x': 40.43}),
    (1.2, {'x': 43.35}),
    (1.3, {'x': 46.17}),
    (1.4, {'x': 48.91}),
    (1.5, {'x': 51.53}),
    (1.6, {'x': 54.05}),
    (1.7, {'x': 56.50}),
    (1.8, {'x': 58.82}),
    (1.9, {'x': 61.05}),
    (2.0, {'x': 63.20}),
    (2.1, {'x': 65.23}),
    (2.2, {'x': 67.17}),
    (2.3, {'x': 69.02}),
    (2.4, {'x': 70.75}),
    (2.5, {'x': 72.40})
] 
```

Now we can start sending the data to the server, checking periodically to see if there is a completed prediction.


```python
from time import sleep

LAST_PREDICTION_TIME = None
for i in range(len(example_data)):
    # Send data to server
    print(f'{example_data[i][0]}s: Sending data to server... ', end='')
    session.send_data(time=example_data[i][0], **example_data[i][1])

    # Check for a prediction result
    status = session.get_prediction_status()
    if LAST_PREDICTION_TIME != status["last prediction"]: 
        # New prediction result
        LAST_PREDICTION_TIME = status["last prediction"]
        print('Prediction Completed')
        
        # Get prediction
        # Prediction is returned as a type uncertain_data, so you can manipulate it like that datatype.
        # See https://nasa.github.io/prog_algs/uncertain_data.html
        t, prediction = session.get_predicted_toe()
        print(f'Predicted ToE (using state from {t}s): ')
        print(prediction.mean)

        # Get Predicted future states
        # You can also get the predicted future states of the model.
        # States are saved according to the prediction configuration parameter 'save_freq' or 'save_pts'
        # In this example we have it setup to save every 1 second.
        # Return type is UnweightedSamplesPrediction (since we're using the monte carlo predictor)
        # See https://nasa.github.io/prog_algs
        t, event_states = session.get_predicted_event_state()
        print(f'Predicted Event States (using state from {t}s): ')
        es_means = [(event_states.times[i], event_states.snapshot(i).mean) for i in range(len(event_states.times))]
        for time, es_mean in es_means:
            print(f"\t{time}s: {es_mean}")

        # Note: you can also get the predicted future states of the model (see get_predicted_states()) or performance parameters (see get_predicted_performance_metrics())

    else:
        print('No prediction yet')
        # No updated prediction, send more data and check again later.
    sleep(0.1)
```

Notice that the prediction wasn't updated every time step. It takes a bit of time to perform a prediction.

*Note*: You can also get the model from prog_server to work with directly.


```python
model = session.get_model()
print(model)
```

### Example: Option Scoring
This example creates a session with the server to run prognostics for a BatteryCircuit. Three options with different loading profiles are compared by creating a session for each option and comparing the resulting prediction metrics.

First step is to prepare load profiles to compare. Each load profile has format `Array[Dict]`. Where each dict is in format {TIME: LOAD}, where TIME is the start of that loading in seconds. LOAD is a dict with keys corresponding to model.inputs.

Note: Dict must be in order of increasing time

Here we introduce 3 load profiles to be used with simulation:


```python
plan0 = {
    0: {'i': 2},
    600: {'i': 1},
    900: {'i': 4},
    1800: {'i': 2},
    3000: {'i': 3}
}
```


```python
plan1 = {
    0: {'i': 3},
    900: {'i': 2},
    1000: {'i': 3.5},
    2000: {'i': 2.5},
    2300: {'i': 3}
}
```


```python
plan2 = {
    0: {'i': 1.25},
    800: {'i': 2},
    1100: {'i': 2.5},
    2200: {'i': 6},
}
```


```python
LOAD_PROFILES = [plan0, plan1, plan2]
```

The next step is to open a session with the battery circuit model for each of the 3 plans. We are specifying a time of interest of 2000 seconds (for the sake of a demo). This could be the end of a mission/session, or some inspection time.


```python
sessions = [
    prog_client.Session(
        'BatteryCircuit',
        pred_cfg = {
            'save_pts': [2000],
            'save_freq': 1e99, 'n_samples':15},
        load_est = 'Variable',
        load_est_cfg = LOAD_PROFILES[i]) 
    for i in range(len(LOAD_PROFILES))]
```

Now let's wait for prognostics to complete


```python
for session in sessions:
    sessions_in_progress = True
    while sessions_in_progress:
        sessions_in_progress = False
        status = session.get_prediction_status()
        if status['in progress'] != 0:
            print(f'\tSession {session.session_id} is still in progress')
            sessions_in_progress = True
            time.sleep(STEP)
    print(f'\tSession {session.session_id} complete')
print('All sessions complete')
```

Now that the sessions are complete, we can get the results.


```python
results = [session.get_predicted_toe()[1] for session in sessions]
```

Now let's compare results. First let's look at the mean Time to Event (ToE):


```python
print('Mean ToE:')
best_toe = 0
best_plan = None
for i in range(len(results)):
    mean_toe = results[i].mean['EOD']
    print(f'\tOption {i}: {mean_toe:0.2f}s')
    if mean_toe > best_toe:
        best_toe = mean_toe
        best_plan = i
print(f'Best option using method 1: Option {best_plan}')
```

As a second metric, let's look at the SOC at our point of interest (2000 seconds)


```python
best_soc = 0
best_plan = None
soc = [session.get_predicted_event_state()[1] for session in sessions]
for i in range(len(soc)):
    mean_soc = soc[i].snapshot(-1).mean['EOD']
    print(f'\tOption {i}: {mean_soc:0.3f} SOC')
    if mean_soc > best_soc:
        best_soc = mean_soc
        best_plan = i
print(f'Best option using method 2: Option {best_plan}')
```

Other metrics can be used as well, like probability of mission success given a certain mission time, uncertainty in ToE estimate, final state at end of mission, etc. 

## Using prog_server - REST Interface

Communication with ProgPy is through a rest interface. The RestAPI is described here: [Rest API](https://app.swaggerhub.com/apis-docs/teubert/prog_server/).

Most programming languages have a way of interacting with REST APIs (either native or through a package/library). `curl` requests can also be used by command line or apps like Postman.

## Custom Models
**A version of this section will be added in release v1.8** 

## Closing
When you're done using prog_server, make sure you turn off the server.


```python
prog_server.stop()
```
