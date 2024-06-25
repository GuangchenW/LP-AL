# Local Penalization Adaptive Learning (LP-AL)

## Description
The project contains the source code used in "Parallelizing Adaptive Reliability Analysis through Penalizaing the Learning Function" (currently under review). The code provides the ability to:
* Automatically generate samples needed based on variable distributions.
* Automate and parallelize the adaptive reliability analysis process.
* Allow direct user interaction with the optimization process through an ask-tell interface.

Example systems and their respective limit-state functions are provided for testing purposes. Five learning functions are also provided. However, more can be implemented by the user. The following figures demonstrates the effectiveness of LP-AL using a simple 2D example.

![Limit state estimation by LP-AL]((https://github.com/GuangchenW/LP-AL/blob/main/additional_figures/4B_U_8.png))
*Estimated limit state and response surface by the LP-AL algorithm using the U learning function when batch size is set to 8. The true limit state function is the four-branch series system described in `objective_functions/G_4B.py`*

![Batch selections by LP-AL](https://github.com/GuangchenW/LP-AL/blob/main/additional_figures/4B_U_8.gif)
*Batch selections by LP-AL using the U learning function when batch size is set to 8. The blue points represents the reduced sample pool and the red points are selected for the current batch. Number at bottom left indicates the number of iterations.*
---

## Requirements
* Project developed with `python3.10.7` on Windows 11. Source code is platform independent but untested on macOS and Linux.
* `virtualenv` is required. See [here](https://virtualenv.pypa.io/en/latest/installation.html) for how to install.

## Setting up
* Run `python -m venv .` in the project root directory to create a python virtual environment.
* With the virtual environment activated, run `pipenv sync` to install all dependencies.

## Extra dependencies
* [GNU Octave](https://octave.org/) is required in order to perform reliability analysis for the `G_FEM` limit-state function.
* `ffmpeg` is recommended for creating animated visuals from plots. `pillow` will be used if `ffmpeg` is unavailable.

## How to use
* A simple overview of the basic usage is given here. For more advanced usage examples, see `run_simulation.py`.
### LP-AL
* The main algorithm is implemented in the `AKMCS` class in `ak_mcs.py`.
* The `acq_func` and `batch_size` arguments determine the learning function and batch size of the LP-AL process. By default, the U learning function is used with batch size 1 (sequential strategy).
* The `initialize_input` function is used to associate a limit-state function with the `AKMCS` object. If the initial training data `bootstrap_inputs` and `bootstrap_outputs` are not supplied, they will be chosen randomly.
* Once the `AKMCS` object is associated with a limit-state function, `kriging_estimate` performs the LP-AL algorithm. `visualize` can be used to visualize the results once the algorithm converges (only works for 2D limit-state functions).

### Limit-state function
* Two plug-n-play classes, `AskTellFunction` and `AnalyticalFunction`, are provided for creating limit-state functions and automatically generating input samples.
* The `AskTellFunction` class is designed for cases where the limit-state function cannot be evaluated through a computer program. The user will be prompted with an input each time the limit-state function needs to be evaluated. The user should enter the output as a floating point number and press enter. 
* The `AnalyticalFunction` class is designed for cases where evaluations of the limit-state function can be automated with a computer program. For example usages, see `examples.py`.

