import os
import math

import cereal.messaging as messaging
from common.numpy_fast import interp
from common.op_params import opParams
from selfdrive.swaglog import cloudlog
from common.realtime import sec_since_boot
from selfdrive.controls.lib.radar_helpers import _LEAD_ACCEL_TAU
from selfdrive.controls.lib.longitudinal_mpc import libmpc_py
from selfdrive.controls.lib.drive_helpers import MPC_COST_LONG
from selfdrive.controls.lib.dynamic_follow import DynamicFollow

LOG_MPC = os.environ.get('LOG_MPC', False)
LON_MPC_STEP = 0.2  # first step is 0.2s


class LongitudinalMpc():
  def __init__(self, mpc_id):
    self.mpc_id = mpc_id

    self.dynamic_follow = DynamicFollow(mpc_id)
    self.setup_mpc()
    self.v_cruise = 0.0
    self.prev_lead_status = False
    self.prev_lead_x = 0.0
    self.new_lead = False

    self.last_cloudlog_t = 0.0
    self.n_its = 0
    self.duration = 0

    self.op_params = opParams()

  def publish(self, pm):
    if LOG_MPC:
      qp_iterations = max(0, self.n_its)
      dat = messaging.new_message('liveLongitudinalMpc')
      dat.liveLongitudinalMpc.xEgo = list(self.mpc_solution[0].x_ego)
      dat.liveLongitudinalMpc.vEgo = list(self.mpc_solution[0].v_ego)
      dat.liveLongitudinalMpc.aEgo = list(self.mpc_solution[0].a_ego)
      dat.liveLongitudinalMpc.xLead = list(self.mpc_solution[0].x_l)
      dat.liveLongitudinalMpc.vLead = list(self.mpc_solution[0].v_l)
      dat.liveLongitudinalMpc.cost = self.mpc_solution[0].cost
      dat.liveLongitudinalMpc.aLeadTau = self.a_lead_tau
      dat.liveLongitudinalMpc.qpIterations = qp_iterations
      dat.liveLongitudinalMpc.mpcId = self.mpc_id
      dat.liveLongitudinalMpc.calculationTime = self.duration
      pm.send('liveLongitudinalMpc', dat)

  def setup_mpc(self):
    ffi, self.libmpc = libmpc_py.get_libmpc(self.mpc_id)
    self.libmpc.init(MPC_COST_LONG.TTC, MPC_COST_LONG.DISTANCE,
                     MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)

    self.mpc_solution = ffi.new("log_t *")
    self.cur_state = ffi.new("state_t *")
    self.cur_state[0].v_ego = 0
    self.cur_state[0].a_ego = 0
    self.a_lead_tau = _LEAD_ACCEL_TAU

  def set_cur_state(self, v, a):
    self.cur_state[0].v_ego = v
    self.cur_state[0].a_ego = a

  def update(self, CS, lead):
    v_ego = CS.vEgo

    # Setup current mpc state
    self.cur_state[0].x_ego = 0.0

    if lead is not None and lead.status:
      x_lead = lead.dRel
      v_lead = max(0.0, lead.vLead)
      a_lead = lead.aLeadK

      if (v_lead < 0.1 or -a_lead / 2.0 > v_lead):
        v_lead = 0.0
        a_lead = 0.0

      self.a_lead_tau = lead.aLeadTau
      self.new_lead = False
      if not self.prev_lead_status or abs(x_lead - self.prev_lead_x) > 2.5:
        des_ts = 0.15  # this init function was given the vel at 0.15 in future from prev solution
        # (0.2 sec timestep - 0.05 sec iteration)
        one_step = self.mpc_solution[0].v_ego[1]
        cur_sol = self.mpc_solution[0].v_ego[0]
        v_mpc = des_ts * (one_step - cur_sol) / LON_MPC_STEP + cur_sol
        self.libmpc.init_with_simulation(v_mpc, x_lead, v_lead, a_lead, self.a_lead_tau)
        self.new_lead = True

      self.dynamic_follow.update_lead(v_lead, a_lead, x_lead, lead.status, self.new_lead)
      self.prev_lead_status = True
      self.prev_lead_x = x_lead
      self.cur_state[0].x_l = x_lead
      self.cur_state[0].v_l = v_lead
    else:
      self.dynamic_follow.update_lead(new_lead=self.new_lead)
      self.prev_lead_status = False
      # Fake a fast lead car, so mpc keeps running
      self.cur_state[0].x_l = 50.0
      self.cur_state[0].v_l = v_ego + 10.0
      a_lead = 0.0
      self.a_lead_tau = _LEAD_ACCEL_TAU

    TR = self.dynamic_follow.update(CS, self.libmpc)  # update dynamic follow

    # Calculate mpc
    t = sec_since_boot()
    self.n_its = self.libmpc.run_mpc(self.cur_state, self.mpc_solution, self.a_lead_tau, a_lead, TR)
    self.duration = int((sec_since_boot() - t) * 1e9)

    # accel_t = self.op_params.get('future_accel_t')  # + 0.15
    # self.a_mpc_future = interp(accel_t, [i / 5 for i in range(11)], self.mpc_solution[0].a_ego[0:11])

    # Reset if NaN or goes through lead car
    crashing = any(lead - ego < -50 for (lead, ego) in zip(self.mpc_solution[0].x_l, self.mpc_solution[0].x_ego))
    nans = any(math.isnan(x) for x in self.mpc_solution[0].v_ego)
    backwards = min(self.mpc_solution[0].v_ego) < -0.01

    if ((backwards or crashing) and self.prev_lead_status) or nans:
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning("Longitudinal mpc %d reset - backwards: %s crashing: %s nan: %s" % (
                          self.mpc_id, backwards, crashing, nans))

      self.libmpc.init(MPC_COST_LONG.TTC, MPC_COST_LONG.DISTANCE,
                       MPC_COST_LONG.ACCELERATION, MPC_COST_LONG.JERK)
      self.cur_state[0].v_ego = v_ego
      self.cur_state[0].a_ego = 0.0
      self.prev_lead_status = False
