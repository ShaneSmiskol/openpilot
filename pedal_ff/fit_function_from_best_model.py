import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from models.konverter.accel_to_gas import predict as best_model_predict
from selfdrive.config import Conversions as CV


def fit_all(x_input, _a3, _a4, _a5, _offset, _e1, _e2, _e3, _e4, _e5, _e6, _e7, _e8):  # , _e1, _e2, _e3, _e4, _e5):  # , _lin1, _lin2):
  """
    x_input is array of a_ego and v_ego
    all _params are to be fit by curve_fit
    kf is multiplier from angle to torque
    c1-c3 are poly coefficients
  """
  a_ego, v_ego = x_input.copy()

  speed_part = (_e5 * a_ego + _e6) * v_ego ** 2 + (_e7 * a_ego + _e8) * v_ego
  accel_part = ((_e1 * v_ego + _e2) * a_ego ** 5 + (_e3 * v_ego + _e4) * a_ego ** 4 + _a3 * a_ego ** 3 + _a4 * a_ego ** 2 + _a5 * a_ego)
  return speed_part + accel_part + _offset

  # # _s1, offset = [((0.011+.02)/2 + .02) / 2, 0.011371989131620245 - .02 - (.016+.0207)/2]
  # _s1, offset = [((0.011 + .02) / 2 + .0155) / 2, 0.011371989131620245 - .02 - (.016 + .0207) / 2]  # these two have been tuned manually since the curve_fit function didn't seem exactly right
  # _a1, _a2, _a3 = [0.022130745681601702, -0.09109186615316711, 0.20997207156680778]
  # speed_part = (_s1 * v_ego)
  #
  # accel_part = (_a1 * a_ego ** 3 + _a2 * a_ego ** 2) * np.interp(v_ego, [10. * CV.MPH_TO_MS, 19. * CV.MPH_TO_MS], [1, 0.6])  # todo make this a linear function and clip (quicker)
  #
  # accel_part += (_a3 * a_ego)
  # accel_part *= np.interp(a_ego, [0, 2], [0.8, 1])
  # # offset -= np.interp(v_ego, [0 * CV.MPH_TO_MS, 6 * CV.MPH_TO_MS], [.04, 0]) * np.interp(a_ego, [0.5, 2], [1, 0])  # np.clip(1 - a_ego, 0, 1)
  # return accel_part + speed_part + offset
  # # return _c1 * v_ego + _c2 * a_ego + _c3


n_samples = 5000

tries = 10000
target_mae = 0.00412991590894452 # 0.004199115616571038
best_params = []
_try = 0
fails = 0
while tries > 0:
  if fails > 5:
    raise Exception('Concurrently failing, function not fittable!')
  _try += 1
  print(f'Try: {_try}')
  tries -= 1

  random_accels = np.random.uniform(-1.1305768489837646, 2.9493775367736816, (n_samples,))
  random_speeds_test = np.random.uniform(0.005201038904488087, 8.493703842163086, (n_samples,))
  # percent_low = 0.9
  # random_speeds_1 = np.random.uniform(0.005201038904488087, 8.493703842163086/2, (round(n_samples*percent_low),))
  # random_speeds_2 = np.random.uniform(8.493703842163086/2, 8.493703842163086, (round(n_samples*(1-percent_low)),))
  # random_speeds = np.array(random_speeds_1.tolist() + random_speeds_2.tolist())
  random_speeds = np.random.uniform(0.005201038904488087, 8.493703842163086, (n_samples,))
  # random_speeds = np.random.uniform(3 * CV.MPH_TO_MS, 6 * CV.MPH_TO_MS, (n_samples,))
  x_train = np.array([random_accels, random_speeds])
  x_test = np.array([random_accels, random_speeds_test])

  predicted_gas = best_model_predict(x_train.T).reshape(-1)
  predicted_gas_test = best_model_predict(x_test.T).reshape(-1)
  try:
    params, covs = curve_fit(fit_all, x_train, predicted_gas)  # , p0=[0.012932506667178914, -0.03919964891667081, 0.1293203179171404, 0.2036552940756803, -0.2718571386226048, 2.379380344735899, 0.03387524338748423, 10. * CV.MPH_TO_MS, 19. * CV.MPH_TO_MS, 0.026995809521699422, 0.032942881750990735])
    fails = 0
  except:
    fails += 1
    continue
  fitted_gas = [fit_all([accel, speed], *params) for accel, speed in x_test.T]
  mae = np.mean(np.abs(np.array(fitted_gas) - predicted_gas_test))
  if mae < target_mae:
    target_mae = mae
    best_params = list(params)
    print('Below target mae: {}'.format(mae))
    break
if len(best_params) == 0:
  raise Exception('Unable to find params better than target mae: {}'.format(target_mae))
params = best_params
print('Fitted vs. function (mae): {}'.format(target_mae))
print(f'Params: {params}')

data = [{'a_ego': a_ego, 'v_ego': v_ego, 'gas': gas} for a_ego, v_ego, gas in zip(random_accels, random_speeds, predicted_gas)]
# poly = np.polyfit(random_accels, predicted_gas, 5)
def compute_gb_new(accel, speed):
  # return np.polyval(poly, accel)
  # params = np.array([0.003837992717277964, -0.01235990011251591, 0.06510535652024786, 0.06600037259754446, -0.0006187306447074457, 0.000597369586548703, 0.0018908153873958748, -0.0004395380613128306, 0.00015113406209297302, 0.0003499560967296682, 0.002631675718307645, 0.0034227193219598844])
  return fit_all([accel, speed], *params)




if ANALYZE_SPEED := True:
  res = 100
  color = 'blue'

  _accels = [
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

    _x_ff = np.linspace(0, 8.9, res)
    _y_ff = [compute_gb_new(np.mean(accel_range), _i) for _i in _x_ff]
    plt.plot(_x_ff * CV.MS_TO_MPH, _y_ff, color='purple', label='new fitted ff function')

    # _y_ff = [model.predict_on_batch(np.array([[np.mean(accel_range), _i]]))[0][0] for _i in _x_ff]
    _y_ff = [best_model_predict(np.array([[np.mean(accel_range), _i]]))[0][0] for _i in _x_ff]
    plt.plot(_x_ff * CV.MS_TO_MPH, _y_ff, color='cyan', label='model ff')
    plt.title(accel_range_str)

    plt.legend()
    plt.xlabel('speed (mph)')
    plt.ylabel('gas')


if ANALYZE_ACCEL := True:
  res = 100

  _speeds = np.r_[[
    [0, 3],
    [3, 6],
    [6, 8],
    [8, 11],
    [11, 14],
    [14, 18],
    [18, 19],
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

    _x_ff = np.linspace(-.6, 2.5, res)
    _y_ff = [compute_gb_new(_i, np.mean(speed_range)) for _i in _x_ff]
    plt.plot(_x_ff, _y_ff, color='purple', label='new fitted ff function')

    _y_ff = [best_model_predict(np.array([[_i, np.mean(speed_range)]]))[0][0] for _i in _x_ff]
    plt.plot(_x_ff, _y_ff, color='cyan', label='model ff')

    plt.legend()
    plt.xlabel('accel (m/s/s)')
    plt.ylabel('gas')
