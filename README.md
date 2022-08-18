# Front Detection

Work in progress. Updated: 18-08-2022

This repository contains a python3 file ('fronts.py') to be used as a module to automatically detect cold, warm and stationary fronts on a range of gridded datasets, and a script that is an example of how the module can be used ('test_front.py'). The module currently relies on inputs being in xarray format.

This code is based on the detection method described in Berry et al. (2011), which itself was based on Hewson (1998). Note that results are more spurious towards the poles, and ideally the code should only be used between 70N and 70S

This code has now been updated to use functions from the Metpy module to calculate the dewpoint and wetbulb temperature.

The module itself has the following dependencies:
- xarray
- numpy
- scipy
- geopy
- metpy

And to plot the data you need: 

- matplotlib
- cartopy


## fronts.py:

Here we will briefly describe each function.
- zeropoints: This finds the zero-crossing points in a gridded data set along the lines of each dimension.

- frontfields: Returns a field where the zero crossings indicate a weather front, along with an indicator of front speed and magnitude.

- linejoin: Turns a list of latitude and longitude points into a list of joined lines.

- smoother: Smooths an input 2-d xarray using a given kernel and a number of passes.

- front: Identifies fronts in the data using the Berry et al. method.

## test_front.py:

- test_front_detection: Sample function on how to detect weather fronts in ERA-5 data using fronts.py.

- test_plot: Sample way to plot and save the data that was calculated from the above function.