#!/usr/bin/env python3
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import sys
from models.konverter.accel_to_gas import predict as best_model_predict

from numpy.random import seed
seed(2147483648)

try:
  from opendbc.can.parser import CANParser
  from tools.lib.logreader import MultiLogIterator
  from tools.lib.route import Route
  from cereal import car
  os.chdir('/openpilot/pedal_ff')
except:
  sys.path.insert(0, 'C:/Git/openpilot-repos/op-smiskol')
  os.environ['PYTHONPATH'] = '.'

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from tqdm import tqdm   # type: ignore
from scipy.optimize import curve_fit
from common.numpy_fast import interp

from selfdrive.config import Conversions as CV
import seaborn as sns
# from tensorflow.keras import layers
# from tensorflow.keras import models
# from tensorflow.keras import activations
# from tensorflow.keras import regularizers
# from tensorflow.keras import optimizers
# import tensorflow as tf
import pickle


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#   tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#   pass

DT_CTRL = 0.01
MIN_SAMPLES = 5 / DT_CTRL  # seconds to frames
MAX_GAS_INTERCEPTOR = 232
MIN_ACC_SPEED = 19 * CV.MPH_TO_MS
TOP_FIT_SPEED = (19 + 3) * CV.MPH_TO_MS


def transform_car_gas(car_gas):  # transorms car gas to gas command equivalent scale
  CAR_GAS_TO_CMD_POLY = [0.86765184, 0.03172896]  # fit using data
  return np.polyval(CAR_GAS_TO_CMD_POLY, car_gas) if car_gas > 0 else 0  # this fixes resting at a constant value for low car_gas


hyperparameter_defaults = dict(
  dropout_1=0,  # 1/15,
  dropout_2=0,  # 1/20,

  dense_1=2,
  dense_2=2,

  optimizer='adadelta',
  batch_size=16,
  learning_rate=1.,
  epochs=1000,
)


def random_chance(percent: int):
  return np.random.randint(0, 100) < percent or percent == 100


def coast_accel(speed):  # given a speed, output coasting acceleration
  # points = [[0.0, 0.538], [1.697, 0.28],
  #           [2.853, -0.199], [3.443, -0.249],
  #           [19.0 * CV.MPH_TO_MS, -0.145]]
  points = [[0.01, 0.0], [.21, .425], [.3107, .535], [.431, .555],  # with no delay todo: OG
            [.777, .438], [1.928, 0.265], [2.66, -0.179],
            [3.336, -0.250], [MIN_ACC_SPEED, -0.145]]

  # points = [[.0, 1.5], [.431, .555],  # with no delay  # todo: ramped up. rm me?
  #           [.777, .438], [1.928, 0.265], [2.66, -0.179],
  #           [3.336, -0.250], [MIN_ACC_SPEED, -0.145]]

  return interp(speed, *zip(*points))


def compute_gb_old(accel, speed):
  # return (accel * 0.5 + (0.05 * (speed / 20 + 1))) * (speed / 25 + 1)
  return float(accel) / 3.0


# def coasting_func(x_input, _c1, _c2, _c3):  # x is speed
#   return _c3 * x_input ** 2 + _c1 * x_input + _c2


def build_model(shape):
  model = models.Sequential()
  model.add(layers.Input(shape=shape))
  model.add(layers.Dense(config.dense_1, layers.LeakyReLU()))
  model.add(layers.Dropout(config.dropout_1))
  model.add(layers.Dense(config.dense_2, layers.LeakyReLU()))
  model.add(layers.Dropout(config.dropout_2))
  model.add(layers.Dense(1))

  # opt = optimizers.Adam(learning_rate=0.001 / 3, amsgrad=True)
  return model

def accel_to_gas_old(a_ego, v_ego, _a3, _a4, _a5, _offset, _e1, _e2, _e3, _e4, _e5, _e6, _e7, _e8):
  speed_part = (_e5 * a_ego + _e6) * v_ego ** 2 + (_e7 * a_ego + _e8) * v_ego
  # accel_part = ((_e1 * v_ego + _e2) * a_ego ** 5 + (_e3 * v_ego + _e4) * a_ego ** 4 + (_e9 * v_ego + _e10) * a_ego ** 3 + (_e11 * v_ego + _e12) * a_ego ** 2 + _a1 * a_ego)
  accel_part = ((_e1 * v_ego + _e2) * a_ego ** 5 + (_e3 * v_ego + _e4) * a_ego ** 4 + _a3 * a_ego ** 3 + _a4 * a_ego ** 2 + _a5 * a_ego)
  ret = speed_part + accel_part + _offset
  return ret


# def fit_all(x_input, _a1, _offset, _e1, _e2, _e3, _e4, _e5, _e6, _e7, _e8, _e9, _e10, _e11, _e12):
def fit_all(x_input, _a3, _a4, _a5, _a6, _a7, _a8, _s1, _s2, _s3, _s4, _offset):
  """
    x_input is array of a_ego and v_ego
    all _params are to be fit by curve_fit
    kf is multiplier from angle to torque
    c1-c3 are poly coefficients
  """
  accel, speed = x_input.copy()

  # if coast >= 0 and a_ego >= 0 and a_ego >= coast:
  #   weight = np.interp(a_ego, [coast, coast * 2], [0, 1])
  #   a_ego = ((a_ego - coast) * (weight) + a_ego * (1-weight))
  def accel_to_gas(a_ego, v_ego):
    speed_part = (_s1 * a_ego + _s2) * v_ego ** 2 + (_s3 * a_ego + _s4) * v_ego
    accel_part = (_a7 * v_ego + _a8) * a_ego ** 4 + (_a3 * v_ego + _a4) * a_ego ** 3 + _a5 * a_ego ** 2 + _a6 * a_ego
    # pitch_part = _p1 * pitch
    ret = speed_part + accel_part + _offset
    return ret

  gas = accel_to_gas(accel, speed)

  if apply_coast_reduction := True:
    coast = coast_accel(speed)
    spread = 0.2
    x = accel - coast  # start at 0 when at coast accel
    if spread > x >= 0:  # ramp up gas output using s-shaped curve from coast accel to coast + spread
      gas *= 1 / (1 + (x / (spread - x)) ** -3) if x != 0 else 0
  return gas

  coast_spread = 0 if speed < 0.555 else 0.05
  gas = accel_to_gas(accel, speed)
  if accel >= coast - coast_spread:
    coast_spread_weight = np.interp(accel, [coast - coast_spread, coast + coast_spread], [0, 1])  # apply to final gas
    gas *= coast_spread_weight


    # threshold = coast * 1.5 if coast > 0 else coast / 2
    # accel_weight = np.interp(a_ego, [coast, threshold], [0, 1])  # weight of original accel
    # a_ego = (a_ego - coast) * (1 - accel_weight) + a_ego * accel_weight


  # gas = accel_to_gas(a_ego, v_ego)
  # gas_at_coast = accel_to_gas(coast, v_ego)
  #
  # if coast <= a_ego:
  #   weight = interp(a_ego, [coast, threshold], [0, 1])
  #   gas = (gas - gas_at_coast) * (1 - weight) + gas * weight

  return gas

  # return (a_ego * _c1 + (_c4 * (v_ego * _c2 + 1))) * (v_ego * _c3 + 1)
  # return _c4 * a_ego + np.polyval([_c1, _c2, _c3], v_ego)  # use this if we think there is a non-linear speed relationship


def known_bad_accel_to_gas(accel, speed):
  poly, accel_coef = [0.00011699240374307696, 0.01634332377590708, -0.0018321108362775451], 0.1166783696247945
  return (poly[0] * speed ** 2 + poly[1] * speed + poly[2]) + (accel_coef * accel)


def known_good_accel_to_gas(desired_accel, speed):
  # _s1, _s2, _a1, _a2, _a3, offset = [0.00011472033606023426, 0.013876213350425473, 0.023210616718880393, -0.09075484756780133, 0.1956935192117445, -0.01095416947337976]
  # speed_part = (_s1 * speed ** 2 + _s2 * speed)  # this can be linear
  # accel_part = (_a1 * desired_accel ** 3 + _a2 * desired_accel ** 2 + _a3 * desired_accel)
  # return accel_part + speed_part + offset

  # _c1, _c2, _c3, _c4 = [0.015332129994618495, -0.013848089187675144, -0.05406226668839383, 0.180209019025656]  # this function is smooth at low speed
  # return (_c1 * speed + _c2) + (_c3 * desired_accel ** 2 + _c4 * desired_accel)  # but didn't give enough gas above 10 mph

  # ----
  # this function was accurate at high speeds and fairly smooth at low speeds but mashed on the gas sometimes
  # _s1, offset = [((0.011 + .02) / 2 + .0155) / 2, 0.011371989131620245 - .02 - (.016 + .0207) / 2]  # these two have been tuned manually since the curve_fit function didn't seem exactly right
  # _a1, _a2, _a3 = [0.022130745681601702, -0.09109186615316711, 0.20997207156680778]
  # speed_part = (_s1 * speed)
  # accel_part = (_a1 * desired_accel ** 3 + _a2 * desired_accel ** 2) * np.interp(speed, [10. * CV.MPH_TO_MS, 19. * CV.MPH_TO_MS], [1, 0.6])  # todo make this a linear function and clip (quicker)
  # accel_part += (_a3 * desired_accel)
  # accel_part *= np.interp(desired_accel, [0, 2], [0.8, 1])
  # # offset -= np.interp(speed, [0 * CV.MPH_TO_MS, 6 * CV.MPH_TO_MS], [.04, 0]) * np.interp(a_ego, [0.5, 2], [1, 0])  # np.clip(1 - a_ego, 0, 1)
  # return accel_part + speed_part + offset
  # ----

  # _c1, _c2, _c3, _c4 = [0.04412016647510183, 0.018224465923095633, 0.09983653162564889, 0.08837909527049172]  # too much gas at low accel but good gas at higher accels
  # return (desired_accel * _c1 + (_c4 * (speed * _c2 + 1))) * (speed * _c3 + 1)

  # params = np.array([-0.07264304340456754, -0.007522016704006004, 0.16234124452228196, 0.0029096574419830296, 1.1674372321165579e-05, -0.008010070095545522, -5.834025253616562e-05, 0.04722441060805912, 0.001887454016549489, -0.0014370672920621269, -0.007577594283906699, 0.01943515032956308])
  # return accel_to_gas_old(desired_accel, speed, *params)  # todo: good
  def accel_to_gas(a_ego, v_ego, _a3, _a4, _a6, _a7, _a8, _a9, _s1, _s2, _s3, _s4, _offset):
    speed_part = (_s1 * a_ego + _s2) * v_ego ** 2 + (_s3 * a_ego + _s4) * v_ego
    accel_part = (_a8 * v_ego + _a9) * a_ego ** 4 + (_a3 * v_ego + _a4) * a_ego ** 3 + _a6 * a_ego ** 2 + _a7 * a_ego
    ret = speed_part + accel_part + _offset
    return ret
  params = [-0.003705030476784167, -0.022559785625973505, -0.006043320774972937, 0.1365573372786136, 0.0015085405555863522, 0.006770616730616903, 0.0009528920381910715, -0.0017151029060025645, 0.003231645268943276, 0.021256307111157384, -0.005451883616806365]  # this one is actuall smooth at low accels, could be lower though for low speeds
  return accel_to_gas(desired_accel, speed, *params)


def load_processed(file_name):
  with open(file_name, 'rb') as f:
    return pickle.load(f)


def get_accel_delay(speed):
  return int(np.interp(speed, [4 * CV.MPH_TO_MS, 8 * CV.MPH_TO_MS], [5, 20]))


def get_accel_delay_coast(speed):
  return int(np.interp(speed, [5 * CV.MPH_TO_MS, 10 * CV.MPH_TO_MS], [1, 20]))


def offset_accel(_data, coast=False):  # todo: offsetting both speed and accel seem to decrease model loss by a LOT. maybe we should just offset all gas instead of these two todo: maybe not?
  for i in range(len(_data)):  # accounts for delay (moves a_ego up by x samples since it lags behind gas)
    # v_ego = [line['v_ego'] for line in _data[i]]
    a_ego = [line['a_ego'] for line in _data[i]]
    data_len = len(_data[i])
    for j in range(data_len):
      accel_delay = get_accel_delay(_data[i][j]['v_ego']) if not coast else get_accel_delay_coast(_data[i][j]['v_ego'])  # interpolate accel delay from speed
      # if j < accel_delay:  # (v_ego)
      #   continue
      # _data[i][j]['v_ego'] = v_ego[j - accel_delay]
      if j + accel_delay >= data_len:  # (a_ego)
        break
      _data[i][j]['a_ego_current'] = float(_data[i][j]['a_ego'])
      _data[i][j]['a_ego'] = float(a_ego[j + accel_delay])
      # _data[i][j]['v_ego'] = float(v_ego[j + accel_delay])
    # _data[i] = _data[i][accel_delay:]  # removes leading samples (v_ego)
    _data[i] = _data[i][:-accel_delay]  # removes trailing samples (a_ego) (uses last accel delay)
  return _data


def load_and_process_rlogs(lrs, file_name):
  data = [[]]

  for lr in lrs:
    engaged, gas_enable, brake_pressed = False, False, False
    v_ego, gas_command, a_ego, steering_angle, gear_shifter = None, None, None, None, None
    a_target, v_target = None, None
    apply_accel = None
    last_time = 0
    can_updated = False
    ned_orientation = []
    ned_orientation_calib = []
    ecef_orientation = []
    ecef_orientation_calib = []

    signals = [
      ("GAS_COMMAND", "GAS_COMMAND", 0),
      ("GAS_COMMAND2", "GAS_COMMAND", 0),
      ("ENABLE", "GAS_COMMAND", 0),
      ("INTERCEPTOR_GAS", "GAS_SENSOR", 0),
      ("INTERCEPTOR_GAS2", "GAS_SENSOR", 0),
      ("GAS_PEDAL", "GAS_PEDAL", 0),
      ("BRAKE_PRESSED", "BRAKE_MODULE", 0),
      ("SPORT_ON", "GEAR_PACKET", 0),
      ("GEAR", "GEAR_PACKET", 0),
    ]
    cp = CANParser("toyota_corolla_2017_pt_generated", signals)

    all_msgs = sorted(lr, key=lambda msg: msg.logMonoTime)

    def Measurement(msmt):
      return {'value': list(msmt.value), 'std': list(msmt.std), 'valid': bool(msmt.valid)}

    for msg in tqdm(all_msgs):
      if msg.which() == 'carState':
        v_ego = msg.carState.vEgo
        a_ego = msg.carState.aEgo
        steering_angle = msg.carState.steeringAngle
        engaged = msg.carState.cruiseState.enabled
        gear_shifter = msg.carState.gearShifter
      elif msg.which() == 'controlsState':
        a_target = msg.controlsState.aTarget
        v_target = msg.controlsState.vTargetLead
      elif msg.which() == 'carControl':
        apply_accel = msg.carControl.actuators.gas - msg.carControl.actuators.brake
      elif msg.which() == 'liveLocationKalman':
        ned_orientation = Measurement(msg.liveLocationKalman.orientationNED)
        ned_orientation_calib = Measurement(msg.liveLocationKalman.orientationNEDCalibrated)
        ecef_orientation = Measurement(msg.liveLocationKalman.orientationECEF)
        ecef_orientation_calib = Measurement(msg.liveLocationKalman.calibratedOrientationECEF)

      if msg.which() not in ['can', 'sendcan']:
        continue
      cp_updated = cp.update_string(msg.as_builder().to_bytes())  # usually all can signals are updated so we don't need to iterate through the updated list

      for u in cp_updated:
        if u == 0x200:  # GAS_COMMAND
          can_updated = True

      gas_enable = bool(cp.vl['GAS_COMMAND']['ENABLE'])
      gas_command = max(round(cp.vl['GAS_COMMAND']['GAS_COMMAND'] / 255., 5), 0.0)  # unscale, round, and clip
      assert gas_command <= 1, "Gas command above 100%, look into this"

      user_gas = (cp.vl['GAS_SENSOR']['INTERCEPTOR_GAS'] + cp.vl['GAS_SENSOR']['INTERCEPTOR_GAS2']) / 2.  # only for user todo: is the max 232?
      car_gas = cp.vl['GAS_PEDAL']['GAS_PEDAL']  # for user AND openpilot/car (less noisy than interceptor but need to check we're not engaged)

      brake_pressed = bool(cp.vl['BRAKE_MODULE']['BRAKE_PRESSED'])
      sport_on = bool(cp.vl['GEAR_PACKET']['SPORT_ON'])

      if msg.which() != 'can':  # only store when can is updated
        continue

      if abs(msg.logMonoTime - last_time) * 1e-9 > 1 / 20:
        print('TIME BREAK!')
        print(abs(msg.logMonoTime - last_time) * 1e-9)

      if (v_ego is not None and can_updated and gear_shifter == car.CarState.GearShifter.drive and not sport_on and  # creates uninterupted sections of engaged data
              abs(msg.logMonoTime - last_time) * 1e-9 < 1 / 20):  # also split if there's a break in time
        data[-1].append({'v_ego': v_ego, 'gas_command': gas_command, 'a_ego': a_ego, 'user_gas': user_gas,
                         'car_gas': car_gas, 'brake_pressed': brake_pressed, 'engaged': engaged, 'gas_enable': gas_enable,
                         'steering_angle': steering_angle, 'a_target': a_target, 'v_target': v_target, 'apply_accel': apply_accel,
                         'ned_orientation': ned_orientation, 'ned_orientation_calib': ned_orientation_calib, 'ecef_orientation': ecef_orientation, 'ecef_orientation_calib': ecef_orientation_calib,
                         'time': msg.logMonoTime * 1e-9})
      elif len(data[-1]):  # if last list has items in it, append new empty section
        data.append([])

      last_time = msg.logMonoTime

  del all_msgs

  print('Max seq. len: {}'.format(max([len(line) for line in data])))

  data = [sec for sec in data if len(sec) > 5 / DT_CTRL]  # long enough sections

  with open(file_name, 'wb') as f:  # now dump
    pickle.dump(data, f)
  return data


def fit_ff_model(use_dir, plot=False):

  if os.path.exists('data'):
    data = load_processed('data')
  else:
    route_dirs = [f for f in os.listdir(use_dir) if '.ini' not in f and f != 'exclude']
    route_files = [[os.path.join(use_dir, i, f) for f in os.listdir(os.path.join(use_dir, i)) if f != 'exclude' and '.ini' not in f] for i in route_dirs]
    lrs = [MultiLogIterator(rd, wraparound=False) for rd in route_files]
    data = load_and_process_rlogs(lrs, file_name='data')

  if os.path.exists('data_coasting'):  # for 2nd function that ouputs decel from speed (assuming coasting)
    data_coasting = load_processed('data_coasting')
  else:
    coast_dir = os.path.join(os.path.dirname(use_dir), 'coast')
    data_coasting = load_and_process_rlogs([MultiLogIterator([os.path.join(coast_dir, f) for f in os.listdir(coast_dir) if '.ini' not in f], wraparound=False)], file_name='data_coasting')

  # def compute_gb_pedal(accel, speed, coast):
  #   def accel_to_gas(a_ego, v_ego):
  #     speed_part = (_e5 * a_ego + _e6) * v_ego ** 2 + (_e7 * a_ego + _e8) * v_ego
  #     # accel_part = ((_e1 * v_ego + _e2) * a_ego ** 5 + (_e3 * v_ego + _e4) * a_ego ** 4 + (_e9 * v_ego + _e10) * a_ego ** 3 + (_e11 * v_ego + _e12) * a_ego ** 2 + _a1 * a_ego)
  #     accel_part = ((_e1 * v_ego + _e2) * a_ego ** 5 + (_e3 * v_ego + _e4) * a_ego ** 4 + _a3 * a_ego ** 3 + _a4 * a_ego ** 2 + _a5 * a_ego)
  #     ret = speed_part + accel_part + _offset
  #     return ret
  #
  #   _a3, _a4, _a5, _offset, _e1, _e2, _e3, _e4, _e5, _e6, _e7, _e8 = [-0.0783068519841404, -0.02425620872221965, 0.13060194634915956, 0.048408210211338176, 5.543874388291277e-05,
  #                                                                     -0.011102981702528086, -0.0003173850406700908, 0.0604232557901408, 0.0012248938828813751, -0.0010763810268259095,
  #                                                                     0.0017804236356551181, 0.011950706937897477]
  #
  #   coast_spread = 0.08
  #   coast_spread_weight = 0
  #   if accel >= coast - coast_spread:
  #     coast_spread_weight = interp(accel, [coast - coast_spread, coast + coast_spread * 2], [0, 1])
  #     # if coast >= 0:
  #     #   accel_weight = 0# interp(accel, [coast - coast_spread, coast + (coast_spread * 3)], [0., 1])
  #     #   print(accel, (coast - coast_spread), accel - (coast - coast_spread))
  #     #   accel = (accel - (coast + coast_spread*2)) * (1 - accel_weight) + accel * accel_weight
  #     #   # print(accel)
  #     #   print()
  #   gas = accel_to_gas(accel, speed) * coast_spread_weight
  #   # if coast >= 0:
  #   # gas_at_coast = accel_to_gas(coast - coast_spread, speed)
  #   # print(gas, gas_at_coast)
  #   # print(accel, coast)
  #   # print()
  #   # gas = gas - (gas_at_coast * np.interp(accel, [coast - coast_spread, 1.5], [1, 0]))
  #
  #   return np.clip(gas, 0., 1.)
  #   # else:
  #   #   return 0.
  #
  #
  # print(len(data))
  # print([len(l) for l in data])
  # data = data[0]
  # data = [l for l in data if l['v_ego'] <= 19 * CV.MPH_TO_MS and l['engaged'] and l['user_gas'] < 15]#[23100:][:700]
  # # plt.plot([l['a_target'] for l in data], label='a_target')
  # plt.plot([l['apply_accel'] * 3 for l in data], label='apply_accel')
  # plt.plot([l['a_ego'] for l in data], label='a_ego')
  # plt.legend()
  # plt.figure()
  # plt.plot([l['v_target'] for l in data], label='v_target')
  # plt.plot([l['v_ego'] for l in data], label='v_ego')
  # plt.legend()
  # plt.figure()
  # plt.plot([l['gas_command'] for l in data], label='gas_command')
  # plt.plot([compute_gb_pedal(l['apply_accel']*3, l['v_ego'], coast_accel(l['v_ego'])) for l in data], label='new gas_command')
  # plt.legend()
  # plt.show()
  # return data, None
  # raise Exception

  # for data_0 in data:
  #   data_0 = [l for l in data_0 if not l['engaged']]
  #   if len(data_0) == 0:
  #     continue
  #   a_ego = np.array([l['a_ego'] for l in data_0])
  #   v_ego = np.array([l['v_ego'] for l in data_0])
  #   gas = np.array([l['car_gas'] for l in data_0])
  #   a_ego = a_ego / np.max(np.abs(a_ego))
  #   # a_ego = (a_ego - a_ego.mean()) / a_ego.std()
  #   # v_ego = (v_ego - v_ego.mean()) / v_ego.std()
  #   # gas = (gas - gas.mean()) / gas.std()
  #   plt.clf()
  #   plt.plot(a_ego, label='a_ego')
  #   plt.plot(np.roll(a_ego, -int(int(get_accel_delay(np.mean([i['v_ego'] * 2.2369 for i in data_0]))))), label='a_ego rolled')
  #   # plt.plot(v_ego, label='v_ego')
  #   plt.plot(gas, label='gas')
  #   plt.title(np.mean([i['v_ego'] * 2.2369 for i in data_0]))
  #   plt.legend()
  #   plt.pause(0.01)
  #   plt.show()
  #   input()
  # raise Exception

  # data = [i for j in data for i in j]  # flatten
  # return data
  #
  # raise Exception

  if OFFSET_ACCEL := True:
    data = offset_accel(data)
  if COAST_OFFSET_ACCEL := False:
    data_coasting = offset_accel(data_coasting, coast=True)

  max_car_gas_fit = max([l['car_gas'] for l in [i for j in data for i in j] if l['engaged'] and l['user_gas'] < 14])
  print('Max fit car_gas for transform_car_gas function: {}'.format(max_car_gas_fit))

  # new_data = []
  # for sec in data:  # remove samples where we're braking in the future but not now
  #   new_sec = []
  #   for idx, line in enumerate(sec):
  #     accel_delay = get_accel_delay(line['v_ego'])  # interpolate accel delay from speed
  #     if idx + accel_delay < len(sec):
  #       if line['brake_pressed'] is sec[idx + accel_delay]['brake_pressed']:
  #         new_sec.append(line)
  #   if len(new_sec) > 0:
  #     new_data.append(new_sec)
  # data = new_data
  # del new_data
  # raise Exception
  #
  # Removes cases where user brakes shortly after giving gas (gas would be positive, accel negative due to accel offsetting)
  # data = [[line for idx, line in enumerate(sec) if (not sec[idx + get_accel_delay(np.mean(i['v_ego'] for i in sec))]['brake_pressed'] if
  #                                                   idx + get_accel_delay(np.mean(i['v_ego'] for i in sec)) < len(sec) else False)] for sec in data]

  data = [i for j in data for i in j]  # flatten
  data_coasting = [i for j in data_coasting for i in j]  # flatten
  print(f'Samples (before filtering): {len(data)}')
  # data += data_coasting

  for line in data:
    if line['engaged'] and line['gas_enable'] and line['gas_command'] > 0.001:  # reduce gas near 0 accel and speed to bias the final function/model
      if line['v_ego'] < 18 * CV.MPH_TO_MS and line['a_ego'] < 1.1:
        # # reduction = np.interp(line['v_ego'], [2 * CV.MPH_TO_MS, 12 * CV.MPH_TO_MS], [1.0, 0])
        # reduction = np.interp(line['a_ego'], [0.8, 1.4], [np.interp(line['v_ego'] * CV.MS_TO_MPH, [2, 10], [1.0, 0]), 0.0])
        # reduction *= 0.08
        reduction = np.interp(line['v_ego'], [1.5 * CV.MPH_TO_MS, 6 * CV.MPH_TO_MS], [1.0, 0])
        reduction *= np.interp(line['a_ego'], [0.25, 1.2], [1, 0])
        reduction *= 0.04
        line['gas_command'] = max(line['gas_command'] - reduction, 0)

        # reduction = np.interp(line['a_ego'], [-0.2, 0.6], [1, 0]) * np.interp(line['v_ego'], [4 * CV.MPH_TO_MS, 8 * CV.MPH_TO_MS, 19 * CV.MPH_TO_MS], [1.0, 0.6, 0.1])
        # line['gas_command'] -= reduction * line['gas_command']

  # Data filtering
  def general_filters(_line):  # general filters
    return 0.01 * CV.MPH_TO_MS < _line['v_ego'] < TOP_FIT_SPEED and not _line['brake_pressed'] and abs(_line['steering_angle']) <= 25

  data_coasting = [line for line in data_coasting if general_filters(line) and line['car_gas'] == 0 and not line['engaged'] and -0.7 < line['a_ego'] < 0.7]

  engaged_samples = 0
  user_samples = 0
  # coast_user = []

  print(f'under 5 mph: {len([l for l in data if l["v_ego"] < 5 * CV.MPH_TO_MS])}')
  print(f'under 5 mph engaged: {len([l for l in data if l["v_ego"] < 5 * CV.MPH_TO_MS and l["engaged"]])}')
  print(f'under 5 mph disengaged: {len([l for l in data if l["v_ego"] < 5 * CV.MPH_TO_MS and not l["engaged"] and l["car_gas"] >= .1])}')
  sns.distplot([l['car_gas'] for l in data if not l['engaged'] and l["v_ego"] < 5 * CV.MPH_TO_MS and l['car_gas'] > 0])

  new_data = []
  for line in data:
    line = line.copy()
    if general_filters(line):
      # since car gas doesn't map to gas command perfectly, only use user samples where gas is above certain threshold
      if not line['engaged'] and line['car_gas'] <= max_car_gas_fit:  # verified for sure working up to 0.65, but probably could go further
        if line['car_gas'] > 0.:
          line['gas'] = float(transform_car_gas(line['car_gas']))  # this matches car gas up with gas cmd fairly accurately
        elif line['user_gas'] < 15 and random_chance(100):
          line['gas'] = 0.
        else:
          continue
        user_samples += 1
      elif line['engaged'] and line['gas_enable'] and line['gas_command'] > 0.001 and line['user_gas'] < 15:  # engaged and user not overriding
        if line['v_ego'] < 2 * CV.MPH_TO_MS and random_chance(35):
          continue
        # # todo this is a hacky fix for bad data. i let op accidentally send gas cmd while not engaged and interceptor didn't like that so it wouldn't apply commanded gas WHILE ENGAGED sometimes. this gets rid of those samples
        diff = abs(line['gas_command'] - transform_car_gas(line['car_gas']))
        if line['car_gas'] == 0 or diff > 0.05:  # function avgs 0.011 error
          # print('SHOULDN\'T BE HERE: {}'.format(diff))
          # print(line)
          continue
        line['gas'] = float(line['gas_command'])
        engaged_samples += 1
      else:
        continue
      line['pitch'] = line['ned_orientation']['value'][1]
      new_data.append(line)

  data = new_data
  del new_data
  print('There are {} engaged samples and {} user samples!'.format(engaged_samples, user_samples))
  print(f'under 5 mph engaged: {len([l for l in data if l["v_ego"] < 5 * CV.MPH_TO_MS and l["engaged"]])}')
  print(f'under 5 mph disengaged: {len([l for l in data if l["v_ego"] < 5 * CV.MPH_TO_MS and not l["engaged"]])}')


  # print('There are {} user coast samples!'.format(len(coast_user)))
  # sns.distplot([line['a_ego'] for line in coast_user], bins=75)

  print(f'under 5 mph: {len([l for l in data if l["v_ego"] < 5 * CV.MPH_TO_MS])}')

  data = [line for line in data if 3. > line['a_ego'] > coast_accel(line['v_ego']) - .6]  # this is experimental
  # data = [line for line in data if line['a_ego'] >= -0.5]  # sometimes a ego is -0.5 while gas is still being applied (todo: maybe remove going up hills? this should be okay for now)
  print(f'Samples (after filtering):  {len(data)}\n')
  print(f'under 5 mph: {len([l for l in data if l["v_ego"] < 5 * CV.MPH_TO_MS])}')


  print(f"Coasting samples: {len(data_coasting)}")

  temp_gas = [l['gas'] for l in data]
  print('Gas min: {} max: {}'.format(round(min(temp_gas), 5), round(max(temp_gas), 5)))
  sns.distplot(temp_gas, bins=75)
  plt.savefig('plots/gas dist.png')

  assert len(data) > MIN_SAMPLES, 'too few valid samples found in route'

  # Now prepare for function fitting
  data_accels = np.array([line['a_ego'] for line in data])
  data_speeds = np.array([line['v_ego'] for line in data])
  # data_pitches = np.array([line['pitch'] for line in data])
  # data_cur_accels = np.array([line['a_ego_current'] for line in data])
  data_gas = np.array([line['gas'] for line in data])
  print('MIN ACCEL: {}'.format(min(data_accels)))
  print(f'accel: {np.min(data_accels), np.max(data_accels)}')
  print(f'speed: {np.min(data_speeds), np.max(data_speeds)}')
  print('Samples below {} mph: {}, samples above: {}'.format(round(MIN_ACC_SPEED * CV.MS_TO_MPH, 2), len([_ for _ in data_speeds if _ < MIN_ACC_SPEED]), len([_ for _ in data_speeds if _ > MIN_ACC_SPEED])))

  x_train = np.array([data_accels, data_speeds]).T
  y_train = np.array(data_gas)

  # model = build_model(x_train.shape[1:])
  # if config.optimizer == 'adam':
  #   opt = optimizers.Adam(learning_rate=config.learning_rate, amsgrad=True)
  # else:
  #   opt = optimizers.Adadelta(learning_rate=config.learning_rate)

  # model.compile(opt, loss='mse', metrics=['mae'])
  # try:
  #   model.fit(x_train, y_train,
  #             batch_size=config.batch_size,
  #             epochs=config.epochs,
  #             validation_split=0.2,
  #             callbacks=[
  #               # tf.keras.callbacks.EarlyStopping('mae', patience=75),
  #               WandbCallback()
  #             ])
  # except KeyboardInterrupt:
  #   print('Training stopped!')
  # exit(0)

  # model = models.load_model('models/model-best.h5')

  # params, covs = curve_fit(fit_all, x_train.T, y_train)
  params = np.array([-0.0034109221270790142, -0.02989942810035373, 0.002005326552420498, 0.1356381902353583, 0.0014222019070588158, 0.008436894892099946, 0.0009439048033890968, -0.0017786568461504919, 0.002986433642380856, 0.021810785976030644, -0.007020501995388009])
  # params = np.array([-0.059593129912793315, -0.04577402603783469, 0.14512438169245576, 0.04943206089162375, 4.829796740317816e-06, -0.009924418151981399, -0.00020755632476486248, 0.052758381188947025, 0.0015371659949146228, -0.001104170167117192, -0.00012929098933089165, 0.011347420516706773])
  print('Params: {}'.format(params.tolist()))
  # params = [((0.011+.02)/2 + .02) / 2, 0.022130745681601702, -0.09109186615316711, 0.20997207156680778, 0.011371989131620245 - .02 - (.016+.0207)/2]

  def compute_gb_new(accel, speed):
    return fit_all([accel, speed], *params)

  from_function = np.array([compute_gb_new(line['a_ego'], line['v_ego']) for line in data])
  print('Fitted function MAE: {}'.format(np.mean(np.abs(data_gas - from_function))))


  if len(data_coasting) > 100:
    print('\nFitting coasting function!')  # (not filtering a_ego gives us more accurate results)
    # coast_params, covs = curve_fit(coast_accel, [line['v_ego'] for line in data_coasting], [line['a_ego'] for line in data_coasting], p0=[0.01, 0.0, 0.21, 0.425, 0.3107, 0.535, 0.431, 0.555, 0.784, 0.443, 1.91, 0.27, 2.809, -0.207, 3.443, -0.249, -0.145])
    # print('Coasting params: {}'.format(coast_params.tolist()))

    # data_coasting_a_ego = np.array([line['a_ego'] for line in data_coasting])
    # from_function = np.array([coasting_func(line['v_ego'], *coast_params) for line in data_coasting])
    # print('Fitted coasting function MAE: {}'.format(np.mean(np.abs(data_coasting_a_ego - from_function))))

    plt.figure()
    plt.title('Coasting data')
    plt.scatter(*zip(*[[line['v_ego'], line['a_ego']] for line in data_coasting]), label='coasting data', s=2)
    x = np.linspace(0, TOP_FIT_SPEED, 1000)
    # plt.plot(x, coasting_func(x, *coast_params))
    # plt.plot(x, coasting_func(x, *coast_params), label='function')

    plt.plot(x, [coast_accel(_x) for _x in x], 'r', label='piecewise function')
    # plt.plot(x, [coast_accel(_x) * 1.5 if coast_accel(_x) > 0 else coast_accel(_x)/2 for _x in x], color='orange', label='threshold')
    # plt.plot(x, [coast_accel(_x) - 0.08 for _x in x], color='orange', label='bounds')
    plt.plot(x, [coast_accel(_x) + 0.2 for _x in x], color='orange')
    plt.plot([0, 8.9], [0, 0])
    plt.legend()
    plt.savefig('imgs/coasting plot.png')
    # raise Exception

    plt.figure()
    x = np.linspace(0, TOP_FIT_SPEED, 100)
    y = [compute_gb_new(coast_accel(spd), spd) for spd in x]  # should be near 0
    plt.plot(x, y)
    plt.legend()
    plt.savefig('imgs/coasting plot-should-be-0.png')
    # raise Exception
  else:
    raise Exception('Not enough coasting samples')

  # if len(params) == 4:
  #   print('FOUND KF: {}'.format(params[0]))
  #   print('FOUND POLY: {}'.format(params[1:].tolist()))
  # elif len(params) == 3:
  #   print('FOUND POLY: {}'.format(params.tolist()))
  # elif len(params) == 1:
  #   print('FOUND KF: {}'.format(params[0]))
  # else:
  #   print('Unsupported number of params')
  #   raise Exception('Unsupported number of params: {}'.format(len(params)))
  # if len(params) > 1 and params[-1] < 0:
  #   print('WARNING: intercept is negative, possibly bad fit! needs more data')
  # print()


  # std_func = []
  # fitted_func = []
  # for line in data:
  #   std_func.append(abs(old_feedforward(line['v_ego'], line['angle_steers']) * old_kf * MAX_TORQUE - line['torque']))
  #   fitted_func.append(abs(CF.get(line['v_ego'], line['angle_steers'], *params) * MAX_TORQUE - line['torque']))
  #
  # print('Torque MAE: {} (standard) - {} (fitted)'.format(np.mean(std_func), np.mean(fitted_func)))
  # print('Torque STD: {} (standard) - {} (fitted)\n'.format(np.std(std_func), np.std(fitted_func)))

  image_suffix = '_5'

  if PLOT_MODEL := True:
    plt.figure()
    plt.clf()
    known_good = [known_good_accel_to_gas(l['a_ego'], l['v_ego']) for l in data]
    # pred = model.predict_on_batch(np.array([[l['a_ego'], l['v_ego']] for l in data])).reshape(-1)
    pred = best_model_predict(np.array([[l['a_ego'], l['v_ego']] for l in data])).reshape(-1)
    fitted_function = [compute_gb_new(l['a_ego'], l['v_ego']) for l in data]

    # print(len(section))
    plt.plot([l['gas'] for l in data], label='gas (ground truth)')
    # plt.plot([l['a_ego'] / 3 for l in data], label='stock output')
    plt.plot(pred, label='model (prediction)')
    # plt.plot(known_good, label='last good')
    plt.plot(fitted_function, label='fitted function')
    plt.legend()
    plt.savefig('plots/model_plot{}.png'.format(image_suffix))
    # raise Exception


  if ANALYZE_SPEED := True:
    plt.figure()
    sns.distplot([line['a_ego'] for line in data], bins=100)
    plt.savefig('plots/accel dist.png')

    res = 100
    color = 'blue'

    _accels = [
      [-.5, 0],
      [-.25, 0],
      [-.5, -.25],
      [0, 0.25],
      [0.25, 0.5],
      [0.4, 0.6],
      [0.5, .75],
      [0.75, 1],
      [1, 1.25],
      [1.25, 1.5],
      [1.5, 1.75],
      [1.75, 2],
      [2, 2.5],
      [2.5, 3],
      [3, 4],
    ]

    for idx, accel_range in enumerate(_accels):
      accel_range_str = '{} m/s/s'.format('-'.join(map(str, accel_range)))
      temp_data = [line for line in data if accel_range[0] <= abs(line['a_ego']) <= accel_range[1]]
      if not len(temp_data):
        continue
      print(f'{accel_range} samples: {len(temp_data)}')
      plt.figure()
      speeds, gas = zip(*[[line['v_ego'], line['gas']] for line in temp_data])
      plt.scatter(np.array(speeds) * CV.MS_TO_MPH, gas, label=accel_range_str, color=color, s=0.05)

      _x_ff = np.linspace(min(speeds), max(speeds), res)

      # _y_ff = [known_bad_accel_to_gas(np.mean(accel_range), _i) for _i in _x_ff]
      # plt.plot(_x_ff * CV.MS_TO_MPH, _y_ff, color='red', label='bad ff function')
      _y_ff = [known_good_accel_to_gas(np.mean(accel_range), _i) for _i in _x_ff]
      plt.plot(_x_ff * CV.MS_TO_MPH, _y_ff, color='green', label='good ff function')

      # _y_ff = [compute_gb_old(np.mean(accel_range), _i) for _i in _x_ff]
      # plt.plot(_x_ff * CV.MS_TO_MPH, _y_ff, color='orange', label='standard ff model at {} m/s/s'.format(np.mean(accel_range)))
      _y_ff = [compute_gb_new(np.mean(accel_range), _i) for _i in _x_ff]
      plt.plot(_x_ff * CV.MS_TO_MPH, _y_ff, color='purple', label='new fitted ff function')

      # _y_ff = [model.predict_on_batch(np.array([[np.mean(accel_range), _i]]))[0][0] for _i in _x_ff]
      _y_ff = [best_model_predict(np.array([[np.mean(accel_range), _i]]))[0][0] for _i in _x_ff]
      plt.plot(_x_ff * CV.MS_TO_MPH, _y_ff, color='cyan', label='model ff')

      plt.legend()
      plt.xlabel('speed (mph)')
      plt.ylabel('gas')
      plt.savefig('plots/s{}{}.png'.format(accel_range_str.replace('/', ''), image_suffix))


  if ANALYZE_ACCEL := True:
    plt.figure()
    sns.distplot([line['v_ego'] for line in data], bins=100)
    plt.savefig('plots/speed dist.png')

    res = 100

    _speeds = np.r_[[
      [0, 0.5],
      [0.5, 1],
      [1, 2],
      [2, 3],
      [3, 6],
      [6, 8],
      [8, 11],
      [11, 14],
      [14, 18],
      [18, 20],
      [20, 22],
      [22, 25],
    ]] * CV.MPH_TO_MS
    color = 'blue'

    for idx, speed_range in enumerate(_speeds):
      speed_range_str = '{} mph'.format('-'.join([str(round(i * CV.MS_TO_MPH, 1)) for i in speed_range]))
      temp_data = [line for line in data if speed_range[0] <= line['v_ego'] <= speed_range[1]]
      if not len(temp_data):
        continue
      print(f'{speed_range_str} samples: {len(temp_data)}')
      plt.figure()
      accels, gas, speeds = zip(*[[line['a_ego'], line['gas'], line['v_ego']] for line in temp_data])
      plt.scatter(accels, gas, label=speed_range_str, color=color, s=0.05)

      plt.plot([coast_accel(np.mean(speed_range)), coast_accel(np.mean(speed_range))], [0, max(gas)], linestyle='--',  color='orange')
      plt.plot([1.5, 1.5], [0, max(gas)], '--', color='orange')

      _x_ff = np.linspace(min(accels), max(accels), res)

      # _y_ff = [known_bad_accel_to_gas(_i, np.mean(speed_range)) for _i in _x_ff]
      # plt.plot(_x_ff, _y_ff, color='red', label='bad ff function')
      _y_ff = [known_good_accel_to_gas(_i, np.mean(speed_range)) for _i in _x_ff]
      plt.plot(_x_ff, _y_ff, color='green', label='good ff function')

      # _y_ff = [compute_gb_old(_i, np.mean(speed_range)) for _i in _x_ff]
      # plt.plot(_x_ff, _y_ff, color='orange', label='standard ff model at {} mph'.format(np.round(np.mean(speed_range) * CV.MS_TO_MPH, 1)))
      _y_ff = [compute_gb_new(_i, np.mean(speed_range)) for _i in _x_ff]
      plt.plot(_x_ff, _y_ff, color='purple', label='new fitted ff function')

      # _y_ff = [model.predict_on_batch(np.array([[_i, np.mean(speed_range)]]))[0][0] for _i in _x_ff]
      _y_ff = [best_model_predict(np.array([[_i, np.mean(speed_range)]]))[0][0] for _i in _x_ff]
      plt.plot(_x_ff, _y_ff, color='cyan', label='model ff')

      plt.legend()
      plt.xlabel('accel (m/s/s)')
      plt.ylabel('gas')
      # plt.xlim(coast_accel(np.mean(speed_range)), 1.5)
      plt.savefig('plots/a{}{}.png'.format(speed_range_str, image_suffix))

  plt.show()

  return data, params

  # if PLOT_3D := False:
  #   X_test = np.linspace(0, max(data_speeds), 20)
  #   Y_test = np.linspace(0, max(data_angles), 20)
  #
  #   Z_test = np.zeros((len(X_test), len(Y_test)))
  #   for i in range(len(X_test)):
  #     for j in range(len(Y_test)):
  #       Z_test[i][j] = CF.get(X_test[i], Y_test[j], *params)
  #
  #   X_test, Y_test = np.meshgrid(X_test, Y_test)
  #
  #   fig = plt.figure()
  #   ax = plt.axes(projection='3d')
  #
  #   surf = ax.plot_surface(X_test * CV.MS_TO_MPH, Y_test, Z_test, cmap=cm.magma,
  #                          linewidth=0, antialiased=False)
  #   fig.colorbar(surf, shrink=0.5, aspect=5)
  #
  #   ax.set_xlabel('speed (mph)')
  #   ax.set_ylabel('angle')
  #   ax.set_zlabel('feedforward')
  #   plt.title('New fitted polynomial feedforward function')


# Compares poly with old ff speed function
# x = np.linspace(0, 30, 100)
# y = x ** 2
# _c1, _c2, _c3 = 0.34365576041121065, 12.845373070976711, 51.63304088261174
# y_poly = _c1 * x ** 2 + _c2 * x + _c3
# plt.plot(x, y_poly, label='poly')
# plt.plot(x, y, label='v_ego**2')
# plt.legend()
# plt.show()


if __name__ == "__main__":
  # r = Route("14431dbeedbf3558%7C2020-11-10--22-24-34")
  # lr = MultiLogIterator(r.log_paths(), wraparound=False)
  use_dir = '/openpilot/pedal_ff/rlogs/use'
  # lr = MultiLogIterator([os.path.join(use_dir, i) for i in os.listdir(use_dir)], wraparound=False)
  data = fit_ff_model(use_dir, plot="--plot" in sys.argv)
