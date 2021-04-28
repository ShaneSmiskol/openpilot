import numpy as np
import math

from common.numpy_fast import interp
from common.op_params import opParams
from selfdrive.swaglog import cloudlog
from common.realtime import sec_since_boot
from selfdrive.controls.lib.longitudinal_mpc_model import libmpc_py

LON_MPC_STEP = 0.2  # first step is 0.2s


class LongitudinalMpcModel():
  def __init__(self):

    self.setup_mpc()
    self.last_cloudlog_t = 0.0
    self.ts = list(range(10))
    self.op_params = opParams()

    self.valid = False

  def setup_mpc(self, v_ego=0.0):
    self.libmpc = libmpc_py.libmpc
    self.libmpc.init(1.0, 1.0, 1.0, 1.0, 1.0)
    self.libmpc.init_with_simulation(v_ego)

    self.mpc_solution = libmpc_py.ffi.new("log_t *")
    self.cur_state = libmpc_py.ffi.new("state_t *")

    self.cur_state[0].x_ego = 0
    self.cur_state[0].v_ego = 0
    self.cur_state[0].a_ego = 0

  def set_cur_state(self, v, a):
    self.cur_state[0].x_ego = 0.0
    self.cur_state[0].v_ego = v
    self.cur_state[0].a_ego = a

  def update(self, v_ego, a_ego, poss, speeds, accels):
    if len(poss) == 0:
      self.valid = False
      return

    x_poly = list(map(float, np.polyfit(self.ts, poss, 3)))
    v_poly = list(map(float, np.polyfit(self.ts, speeds, 3)))
    a_poly = list(map(float, np.polyfit(self.ts, accels, 3)))

    # Calculate mpc
    self.libmpc.run_mpc(self.cur_state, self.mpc_solution, x_poly, v_poly, a_poly)

    self.valid = True

    # Reset if NaN or goes through lead car
    nans = any(math.isnan(x) for x in self.mpc_solution[0].v_ego)

    t = sec_since_boot()
    if nans:
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning("Longitudinal model mpc reset - backwards")

      self.libmpc.init(1.0, 1.0, 1.0, 1.0, 1.0)
      self.libmpc.init_with_simulation(v_ego)

      self.cur_state[0].v_ego = v_ego
      self.cur_state[0].a_ego = 0.0

      self.valid = False
