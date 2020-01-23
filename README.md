**Update: In the future, I will make new branches for new versions of openpilot and push all new commits to that branch (ex. `stock_additions-072`). I will occasionally merge these changes to the main branch (`stock_additions`) once I make sure there are no errors. So to ensure an up to date experience, checkout and drive on the main `stock_additions` branch which will be updated periodically with safe and verified changes! Run this command on your EON to set everything up:**

`cd /data/openpilot;git fetch;git pull;git checkout stock_additions;git reset --hard origin/stock_additions;git checkout stock_additions;git clean -xfd;git pull`

Then run `git branch` to make sure you are on `stock_additions`. The command above is just a one time process for users that have cloned the branch in the past and resolves conflicts.

Shane's Stock Additions 0.7.1 (version 0.1)
=====

This branch is simply stock openpilot with some additions to help it drive as smooth as possible on my 2017 Toyota Corolla w/ comma pedal.


Highlight Features
=====

* [**Dynamic gas**](#dynamic-gas)
* [**Dynamic follow**](#dynamic-follow)
* [**Dynamic lane speed (new!)**](#dynamic-lane-speed)
* [**(NOT YET ADDED) Two PID loops to control gas and brakes independently**](#Two-PID-loops-to-control-gas-and-brakes-independently)
* [**Custom wheel offset to reduce lane hugging**](#Custom-wheel-offset-to-reduce-lane-hugging)
* [**Custom following distance**](#Custom-following-distance)
* [**Customize this branch (opEdit Parameter class)**](#Customize-this-branch-opEdit-Parameter-class)
* [**Live tuning support**](#Live-tuning-support)

-----

Dynamic gas
-----
Currently supported vehicles (w/ comma pedal only):
  * 2017 Toyota Corolla (non-TSS2)
  * Toyota RAV4 (non-TSS2)

This aims to provide a smoother driving experience in stop and go traffic (under 20 mph) by modifying the maximum gas that can be applied based on your current velocity and the relative velocity of the lead car. It'll also of course increase the maximum gas when the lead is accelerating to help you get up to speed quicker than stock. And smoother; this eliminates the jerking you get from stock openpilot with comma pedal. It tries to coast if the lead is only moving slowly, it doesn't use maximum gas as soon as the lead inches forward :). When you are above 20 mph, relative velocity and the following distance is taken into consideration.

Dynamic follow
-----
This is my dynamic follow from 0.5, where it changes your TR (following distance) dynamically based on multiple vehicle factors, as well as data from the lead vehicle. [Here's an old write up from a while ago explaining how it works exactly. Some of it might be out of date, but how it functions is the same.](https://github.com/ShaneSmiskol/openpilot/blob/dynamic-follow/README.md) The goal is to essentially smoothen the driving experience and increase safety, braking sooner.

Dynamic lane speed
-----
*This feature is disabled until I can figure out how to improve it.*

This is a new feature that reduces your cruising speed if many vehicles around you are significantly slower than you. This works with and without an openpilot-identified lead. Ex.: It will slow you down if traveling in an open lane with cars in adjacent lanes that are slower than you. Or if the lead in front of the lead is slowing down, as well as cars in other lanes far ahead. The most it will slow you down is some average of: (the set speed and the average of the surrounding cars) The more the radar points, the more weight goes to the speeds of surrounding vehicles.

~~Two PID loops to control gas and brakes independently~~
-----
***Update**: Probably going to remove this addition, as tuning the current pedal parameters will be a more robust solution in the long run.*

If you have a Toyota Corolla with a comma pedal, you'll love this addition. Two longitudinal PID loops are set up in `longcontrol.py` so that one is running with comma pedal tuning to control the gas, and the other is running stock non-pedal tuning for better braking control. In the car, this feels miles better than stock openpilot, and nearly as good as your stock Toyota cruise control before you pulled out your DSU! It won't accelerate up to stopped cars and brake at the last moment anymore.

Custom wheel offset to reduce lane hugging
-----
***Update**: The performance of this modification is iffy at best, but it definitely is more apparant than just tuning your camera offset value. Removing the immediate angle offset can have some weird oscilating effect when it's windy or on roads with camber (slant to one side). Will be left in, but disabled.*

Stock openpilot doesn't seem to be able to identify your car's true angle offset. With the `LaneHugging` module you can specify a custom angle offset to be added to your desired steering angle. Simply find the angle your wheel is at when you're driving on a straight highway. By default, this is disabled, to enable you can:
- Use the `opEdit` class in the root directory of openpilot. To use it, simply open an `ssh` shell and enter the commands below:
    ```python
    cd /data/openpilot
    python op_edit.py
    ```
    You'll be greeted with a list of your parameters you can explore, enter the number corresponding to `lane_hug_direction`. Your options are to enter `'left'` or `'right'` for whichever direction your car has a tendency to hug toward. `None` will disable the feature.
    Finally you'll need to enter your absolute angle offset (negative will be converted to positive) with the `opParams` parameter: `lane_hug_angle_offset`.

Custom following distance
-----
Using the `following_distance` parameter in `opParams`, you can specify a custom TR value to always be used. Afraid of technology and want to give yourself the highest following distance out there? Try out 2.7s! Are you daredevil and don't care about pissing off the car you're tailgating ahead? Try 0.9s! Please note dynamic follow modifications will be disabled if you set this parameter.
- Again, you can use `opEdit` to change this:
    ```python
    cd /data/openpilot
    python op_edit.py
    ```
    Then enter the number for the `following_distance` parameter and set to a float or integer between `0.9` and `2.7`. `None` will use dynamic follow!

Customize this branch (opEdit Parameter class)
-----
This is a handy tool to change your `opParams` parameters without diving into any json files or code. You can specify parameters to be used in any fork's operation that supports `opParams`. First, ssh in to your EON and make sure you're in `/data/openpilot`, then start `opEdit`:
```python
cd /data/openpilot
python op_edit.py
```
A list of parameters that you can modify are located [here](common/op_params.py#L42).

Parameters are stored at `/data/op_params.json`

Live tuning support
-----
This has just been added and currently only the `camera_offset` and `lane_hug_angle_offset` parameters are supported.
- Just start opEdit with the instructions above and pick a parameter. It will let you know if it supports live tuning, if so, updates will take affect within 5 seconds!
- Alternatively, you can use the new opTune module to live tune quicker and easier! It stays in the parameter edit view so you can more easily experiment with values. opTune show below:

<img src="gifs/op_tune.gif?raw=true" width="600">
