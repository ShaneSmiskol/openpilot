from cereal import car
from common.numpy_fast import clip, interp
from selfdrive.car import apply_toyota_steer_torque_limits, create_gas_command, make_can_msg
from selfdrive.car.toyota.toyotacan import create_steer_command, create_ui_command, \
                                           create_accel_command, create_acc_cancel_command, \
                                           create_fcw_command
from selfdrive.car.toyota.values import Ecu, CAR, STATIC_MSGS, NO_STOP_TIMER_CAR, CarControllerParams, MIN_ACC_SPEED
from opendbc.can.packer import CANPacker
from common.op_params import opParams
from selfdrive.config import Conversions as CV
# from selfdrive.accel_to_gas import predict as accel_to_gas

VisualAlert = car.CarControl.HUDControl.VisualAlert


def accel_hysteresis(accel, accel_steady, enabled):

  # for small accel oscillations within ACCEL_HYST_GAP, don't change the accel command
  if not enabled:
    # send 0 when disabled, otherwise acc faults
    accel_steady = 0.
  elif accel > accel_steady + CarControllerParams.ACCEL_HYST_GAP:
    accel_steady = accel - CarControllerParams.ACCEL_HYST_GAP
  elif accel < accel_steady - CarControllerParams.ACCEL_HYST_GAP:
    accel_steady = accel + CarControllerParams.ACCEL_HYST_GAP
  accel = accel_steady

  return accel, accel_steady
op_params = opParams()

def coast_accel(speed):  # given a speed, output coasting acceleration
  points = [[0.01, op_params.get('0_coast_accel')], [.21, .425], [.3107, .535], [.431, .555],  # with no delay
            [.777, .438], [1.928, 0.265], [2.66, -0.179],
            [3.336, -0.250], [MIN_ACC_SPEED, -0.145]]
  # points = [[.0, op_params.get('0_coast_accel')], [.431, .555],  # with no delay
  #           [.777, .438], [1.928, 0.265], [2.66, -0.179],
  #           [3.336, -0.250], [MIN_ACC_SPEED, -0.145]]
  return interp(speed, *zip(*points))


# def compute_gb_pedal(accel, speed, coast, which_func):
#   # return accel_to_gas([accel, speed])[0]
#   if which_func == 0:
#     _a3, _a4, _a5, _offset, _e1, _e2, _e3, _e4, _e5, _e6, _e7, _e8 = [-0.05340517679465475, -0.053495524853591506, 0.1692496880860915, 0.02378568771700229, 5.9774819503946536e-05, -0.009988274638231051, -0.0003203880816858484, 0.051321083586716484, 0.0023280402254177005, -0.0018446446967463183, -0.008536402106750801, 0.020362858606493128]
#   elif which_func == 1:
#     _a3, _a4, _a5, _offset, _e1, _e2, _e3, _e4, _e5, _e6, _e7, _e8 = [-0.06339414227380791, -0.04252182325129261, 0.16496700047827587, 0.02163184652060803, -7.819573625765886e-05, -0.009744998610583465, 8.451435790895213e-05, 0.05286121507616371, 0.002001649470366154, -0.0019889859912089877, -0.006508368592432109, 0.02155006142761783]
#   elif which_func == 2:
#     _a3, _a4, _a5, _offset, _e1, _e2, _e3, _e4, _e5, _e6, _e7, _e8 = [-0.05594094062330375, -0.048062873804399726, 0.16403002481440093, 0.022714205305227296, -5.7173431029146585e-06, -0.009542845370903873, -0.0001208699128506703, 0.050350551862265606, 0.0020345465756747383, -0.0019496194481763256, -0.006443694337832591, 0.021190635052137256]
#   elif which_func == 3:
#     _a3, _a4, _a5, _offset, _e1, _e2, _e3, _e4, _e5, _e6, _e7, _e8 = [-0.061649360532346216, -0.004917289926341796, 0.15355568143854717, -0.005072052411820398, 0.00019662217949411142, -0.007342717402517755, -0.0005688909172110014, 0.041480849002471086, 0.001880822114313993, -0.0014727057513696277, -0.0071366447268704555, 0.020673978167611646]
#   elif which_func == 4:
#     _a3, _a4, _a5, _offset, _e1, _e2, _e3, _e4, _e5, _e6, _e7, _e8 = [-0.07264304340456754, -0.007522016704006004, 0.16234124452228196, 0.0029096574419830296, 1.1674372321165579e-05, -0.008010070095545522, -5.834025253616562e-05, 0.04722441060805912, 0.001887454016549489, -0.0014370672920621269, -0.007577594283906699, 0.01943515032956308]
#   elif which_func == 5:
#     _a3, _a4, _a5, _offset, _e1, _e2, _e3, _e4, _e5, _e6, _e7, _e8 = [-0.06713195658855746, -0.023450527717881073, 0.1745896831920949, -0.007296857981847622, 7.236149432069842e-05, -0.009064968448433007, -0.00022193411610820104, 0.04985709418327566, 0.002171840642000584, -0.0019409690646680923, -0.010695641108741443, 0.02544971629177782]
#   elif which_func == 6:
#     _a3, _a4, _a5, _offset, _e1, _e2, _e3, _e4, _e5, _e6, _e7, _e8 = [-0.07650987241491732, -0.01313506396806189, 0.18263091764989609, -0.014502272327707318, -5.820955538936409e-05, -0.008699449190029246, 0.00017672175610891123, 0.05076145276583945, 0.002276624696779062, -0.001990422687692836, -0.012951649094935302, 0.026524904498754612]
#   else:
#     _a1, _offset, _e1, _e2, _e3, _e4, _e5, _e6, _e7, _e8, _e9, _e10, _e11, _e12 = [0.051928027407770125, -0.017021340933399423, -0.004932336090691233, 0.012117573094709328, 0.024127324154840687, -0.05483055648050864, 0.002061550701289086, -0.002374642289248953, 0.012143971642267916, 0.029754476475536706, -0.029432879500629568, 0.05542071632500463, -0.00888044738417222, 0.0399769546754697]
#     speed_part = (_e5 * accel + _e6) * speed ** 2 + (_e7 * accel + _e8) * speed
#     accel_part = ((_e1 * speed + _e2) * accel ** 5 + (_e3 * speed + _e4) * accel ** 4 + (_e9 * speed + _e10) * accel ** 3 + (_e11 * speed + _e12) * accel ** 2 + _a1 * accel)
#     return speed_part + accel_part + _offset
#
#   speed_part = (_e5 * accel + _e6) * speed ** 2 + (_e7 * accel + _e8) * speed
#   accel_part = ((_e1 * speed + _e2) * accel ** 5 + (_e3 * speed + _e4) * accel ** 4 + _a3 * accel ** 3 + _a4 * accel ** 2 + _a5 * accel)
#   ret = speed_part + accel_part + _offset
#
#   if coast > 0:
#     weight = interp(accel, [coast / 2, coast * 2], [0, 1.0])
#     apply_accel = (accel - coast) * weight + accel * (1 - weight)
#
#   # ret *= interp(speed, [0, 5 * CV.MPH_TO_MS], [0.75, 1]) * interp(accel, [0, 1], [0.75, 1])
#   return ret
#
#   # # _c1, _c2, _c3, _c4 = [0.04412016647510183, 0.018224465923095633, 0.09983653162564889, 0.08837909527049172]
#   # # return (desired_accel * _c1 + (_c4 * (speed * _c2 + 1))) * (speed * _c3 + 1)
#   # if which_func == 0:
#   #   _c1, _c2, _c3, _c4  = [0.014834278942078814, -0.019486618189634007, -0.04866680885268496, 0.18130227709359556]  # fit on both engaged and disengaged
#   # elif which_func == 1:
#   #   _c1, _c2, _c3, _c4  = [0.015545494731421215, -0.011431576758264202, -0.056374605760840496, 0.1797404798536819]  # just fit on engaged
#   # else:
#   #   _c1, _c2, _c3, _c4, _c5  = [0.0004504646112499155, 0.010911174152383137, 0.020950462773718394, 0.0971672107576878, -0.007383724106218966]
#   #   return (_c1 * speed ** 2 + _c2 * speed + _c5) + (_c3 * desired_accel ** 2 + _c4 * desired_accel)
#   # return (_c1 * speed + _c2) + (_c3 * desired_accel ** 2 + _c4 * desired_accel)


class CarController():
  def __init__(self, dbc_name, CP, VM):
    self.last_steer = 0
    self.accel_steady = 0.
    self.alert_active = False
    self.last_standstill = False
    self.standstill_req = False
    self.op_params = opParams()
    self.standstill_hack = self.op_params.get('standstill_hack')

    self.steer_rate_limited = False

    self.fake_ecus = set()
    if CP.enableCamera:
      self.fake_ecus.add(Ecu.fwdCamera)
    if CP.enableDsu:
      self.fake_ecus.add(Ecu.dsu)

    self.packer = CANPacker(dbc_name)

  def compute_gb_pedal(self, accel, speed, braking, actual_accel):
    def accel_to_gas(a_ego, v_ego):
      speed_part = (_s1 * a_ego + _s2) * v_ego ** 2 + (_s3 * a_ego + _s4) * v_ego
      accel_part = (_a3 * v_ego + _a4) * a_ego ** 3 + (_a5 * v_ego + _a6) * a_ego ** 2 + _a7 * a_ego
      ret = speed_part + accel_part + _offset
      return ret

    _a3, _a4, _a5, _a6, _a7, _s1, _s2, _s3, _s4, _offset = [0.008695917086843731, -0.015614254828356712, -0.02079982940031632, 0.005221453888746468, 0.10533890343060712, 0.0013891995056737531, -0.001231083282494525, 0.009798923905618475, 0.014437517069650387, 0.03717726005999608]
    coast = coast_accel(speed)
    gas = 0.
    coast_spread = self.op_params.get('coast_spread')
    if not braking or accel - self.op_params.get('max_accel_gap') > actual_accel:  # if car not braking or gap between desired accel and actual is too high
      gas = accel_to_gas(accel, speed)
      if self.op_params.get('coast_smoother'):
        gas *= interp(accel, [coast, coast + coast_spread * 2], [0, 1])
    return gas

    # gas = accel_to_gas(accel, speed)
    # if accel >= coast - coast_spread:
    #   coast_spread_weight = interp(accel, [coast - coast_spread, coast + coast_spread], [0, 1])
    #   return clip(gas * coast_spread_weight, 0., 1.)
    # else:
    #   return 0.

    # gas = accel_to_gas(accel, speed)
    # gas_at_coast = accel_to_gas(coast, speed)
    #
    # if coast > 0:
    #   weight = interp(accel, [coast, coast * 1.5], [0.5, 1])
    #   gas = (gas - gas_at_coast) * (1 - weight) + gas * weight

    # return gas

  def update(self, enabled, CS, frame, actuators, pcm_cancel_cmd, hud_alert,
             left_line, right_line, lead, left_lane_depart, right_lane_depart):

    # *** compute control surfaces ***

    # gas and brake
    apply_gas = 0.
    apply_accel = (actuators.gas - actuators.brake) * CarControllerParams.ACCEL_SCALE

    if CS.CP.enableGasInterceptor and enabled and CS.out.vEgo < MIN_ACC_SPEED and self.op_params.get('convert_accel_to_gas'):
      # converts desired acceleration to gas percentage for pedal
      # +0.06 offset to reduce ABS pump usage when applying very small gas
      # apply_accel *= CarControllerParams.ACCEL_SCALE
      apply_gas = self.compute_gb_pedal(apply_accel, CS.out.vEgo, CS.out.brakeLights, CS.out.aEgo)
      if apply_accel > 0 and CS.out.vEgo <= CS.CP.minSpeedCan:  # artifically increase accel to release brake quicker
        apply_accel *= CarControllerParams.ACCEL_SCALE

    # apply_accel, self.accel_steady = accel_hysteresis(apply_accel, self.accel_steady, enabled)
    apply_accel = clip(apply_accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX)
    apply_gas = clip(apply_gas, 0., 1.)

    if enabled and self.op_params.get('apply_gas') is not None:
      apply_gas = self.op_params.get('apply_gas')
      apply_accel = 0


    # steer torque
    new_steer = int(round(actuators.steer * CarControllerParams.STEER_MAX))
    apply_steer = apply_toyota_steer_torque_limits(new_steer, self.last_steer, CS.out.steeringTorqueEps, CarControllerParams)
    self.steer_rate_limited = new_steer != apply_steer

    # Cut steering while we're in a known fault state (2s)
    if not enabled or CS.steer_state in [9, 25] or (abs(CS.out.steeringRateDeg) > 100 and self.op_params.get('steer_fault_fix')):
      apply_steer = 0
      apply_steer_req = 0
    else:
      apply_steer_req = 1

    if not enabled and CS.pcm_acc_status:
      # send pcm acc cancel cmd if drive is disabled but pcm is still on, or if the system can't be activated
      pcm_cancel_cmd = 1

    # on entering standstill, send standstill request
    if CS.out.standstill and not self.last_standstill and CS.CP.carFingerprint not in NO_STOP_TIMER_CAR and not self.standstill_hack:
      self.standstill_req = True
    if CS.pcm_acc_status != 8:
      # pcm entered standstill or it's disabled
      self.standstill_req = False

    self.last_steer = apply_steer
    self.last_accel = apply_accel
    self.last_standstill = CS.out.standstill

    can_sends = []

    #*** control msgs ***
    #print("steer {0} {1} {2} {3}".format(apply_steer, min_lim, max_lim, CS.steer_torque_motor)

    # toyota can trace shows this message at 42Hz, with counter adding alternatively 1 and 2;
    # sending it at 100Hz seem to allow a higher rate limit, as the rate limit seems imposed
    # on consecutive messages
    if Ecu.fwdCamera in self.fake_ecus:
      can_sends.append(create_steer_command(self.packer, apply_steer, apply_steer_req, frame))

      # LTA mode. Set ret.steerControlType = car.CarParams.SteerControlType.angle and whitelist 0x191 in the panda
      # if frame % 2 == 0:
      #   can_sends.append(create_steer_command(self.packer, 0, 0, frame // 2))
      #   can_sends.append(create_lta_steer_command(self.packer, actuators.steeringAngleDeg, apply_steer_req, frame // 2))

    # we can spam can to cancel the system even if we are using lat only control
    if (frame % 3 == 0 and CS.CP.openpilotLongitudinalControl) or (pcm_cancel_cmd and Ecu.fwdCamera in self.fake_ecus):
      lead = lead or CS.out.vEgo < 12.    # at low speed we always assume the lead is present do ACC can be engaged

      # Lexus IS uses a different cancellation message
      if pcm_cancel_cmd and CS.CP.carFingerprint == CAR.LEXUS_IS:
        can_sends.append(create_acc_cancel_command(self.packer))
      elif CS.CP.openpilotLongitudinalControl:
        can_sends.append(create_accel_command(self.packer, apply_accel, pcm_cancel_cmd, self.standstill_req, lead))
      else:
        can_sends.append(create_accel_command(self.packer, 0, pcm_cancel_cmd, False, lead))

    if (frame % 2 == 0) and (CS.CP.enableGasInterceptor):
      # send exactly zero if apply_gas is zero. Interceptor will send the max between read value and apply_gas.
      # This prevents unexpected pedal range rescaling
      can_sends.append(create_gas_command(self.packer, apply_gas, frame//2))

    # ui mesg is at 100Hz but we send asap if:
    # - there is something to display
    # - there is something to stop displaying
    fcw_alert = hud_alert == VisualAlert.fcw
    steer_alert = hud_alert == VisualAlert.steerRequired

    send_ui = False
    if ((fcw_alert or steer_alert) and not self.alert_active) or \
       (not (fcw_alert or steer_alert) and self.alert_active):
      send_ui = True
      self.alert_active = not self.alert_active
    elif pcm_cancel_cmd:
      # forcing the pcm to disengage causes a bad fault sound so play a good sound instead
      send_ui = True

    if (frame % 100 == 0 or send_ui) and Ecu.fwdCamera in self.fake_ecus:
      can_sends.append(create_ui_command(self.packer, steer_alert, pcm_cancel_cmd, left_line, right_line, left_lane_depart, right_lane_depart))

    if frame % 100 == 0 and Ecu.dsu in self.fake_ecus:
      can_sends.append(create_fcw_command(self.packer, fcw_alert))

    #*** static msgs ***

    for (addr, ecu, cars, bus, fr_step, vl) in STATIC_MSGS:
      if frame % fr_step == 0 and ecu in self.fake_ecus and CS.CP.carFingerprint in cars:
        can_sends.append(make_can_msg(addr, vl, bus))

    return can_sends
