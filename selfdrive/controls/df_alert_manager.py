import cereal.messaging as messaging
from selfdrive.controls.lib.dynamic_follow.support import dfProfiles
from common.realtime import sec_since_boot


class dfAlertManager:
  def __init__(self, op_params, is_df=False):
    self.op_params = op_params
    self.is_df = is_df
    self.df_profiles = dfProfiles()
    self.sm = messaging.SubMaster(['dynamicFollowButton', 'dynamicFollowData'])
    self.current_profile = self.df_profiles.to_idx[self.op_params.get('dynamic_follow', default='relaxed').strip().lower()]
    self.prediction_profile = 0
    self.alert_duration = 2.0

    self.offset = None
    self.profile_pred = None
    self.last_button_status = 0
    self.change_time = sec_since_boot()

  @property
  def is_auto(self):
    return self.current_profile == self.df_profiles.auto

  def update(self):
    self.sm.update(0)
    changed = False
    if self.offset is None:
      changed = True
      self.offset = self.current_profile  # ensure we start at the user's current profile
    else:
      status = self.sm['dynamicFollowButton'].status
      new_profile = (status + self.offset) % len(self.df_profiles.to_profile)
      if self.last_button_status != status:
        self.change_time = sec_since_boot()
        changed = True
        self.op_params.put('dynamic_follow', new_profile)  # save current profile for next drive
        self.current_profile = new_profile
        self.last_button_status = status
      elif self.is_auto:
        profile_pred = self.sm['dynamicFollowData'].profilePred
        changed = self.prediction_profile != profile_pred and sec_since_boot() - self.change_time > self.alert_duration
        self.prediction_profile = profile_pred
        return self.prediction_profile, changed, self.change_time

    return self.current_profile, changed, self.change_time
