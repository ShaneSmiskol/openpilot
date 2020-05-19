import matplotlib.pyplot as plt
import numpy as np
import math

res = 50

printed = False
def _print(msg):
  global printed
  if not printed:
    print(msg)
    printed = True


x = np.array([-2.6822, -1.7882, -0.8941, -0.447, -0.2235, 0.0, 0.2235, 0.447, 0.8941, 1.7882, 2.6822]) * 2.2369
# y = [0.35, 0.3, 0.125, 0.09375, 0.075, 0, -0.09, -0.09375, -0.125, -0.3, -0.35]
y = [0.35, 0.3, 0.125, 0.09375, 0.075, 0, -0.075, -0.09375, -0.125, -0.3, -0.35]


v_ego_start = 0
v_ego_end = 5

v_lead_start = 0
v_lead_end = 6
time = 2

v_egos = np.linspace(v_ego_start, v_ego_end, res)
# a_lead = 10
v_leads = np.linspace(v_lead_start, v_lead_end, res)


TR = 1.8

v_rels = [v_lead - v_ego for v_lead, v_ego in zip(v_leads, v_egos)]
rel_accels = [(v_rel - v_rels[0]) / time for v_rel in v_rels]
rel_accels2 = [(v_rels[0] - v_rel) / time for v_rel in v_rels]



v_ego_change = v_egos[-1] - v_egos[0]
v_lead_change = v_leads[-1] - v_leads[0]
print('v_ego_change: {}'.format(v_ego_change))
print('v_lead_change: {}'.format(v_lead_change))

print('range: 1 to 2')
rel_accels3 = []
for v_lead, v_ego in zip(v_leads, v_egos):
  p1 = (v_lead - v_rels[0]) / time

  # print(v_lead-v_ego)
  # lst = [v_lead_start-v_ego_start, v_lead-v_ego], [v_ego_start/(v_ego_start+v_lead_start), v_ego-v_lead]
  # lst = [v_lead_start-v_ego_start, (v_lead-v_ego)*2], [v_lead_start/(v_ego_start+v_lead_start), v_lead_end/(v_ego_end+v_lead_end)]
  # lst_1 = [v_lead_start-v_ego_start, (v_lead_start-v_ego_start)*2]
  lst_1 = [v_lead_start-v_ego_start, v_lead_end-v_ego_end]
  lst_1.insert(1, (lst_1[0] + lst_1[1]) / 2)
  lst_2 = [round((v_lead - v_ego)/v_lead, 3), round((v_leads[len(v_leads) // 2] - v_egos[len(v_leads) // 2])/v_egos[len(v_leads) // 2], 3), round((v_lead - v_ego)/v_lead, 3)]
  print(lst_1, lst_2)
  # print(lst)
  mod = np.interp(v_lead-v_ego, lst_1, lst_2)
  print(mod)
  if v_ego > v_lead and mod < 0:
    mod = abs(mod)
    mod = -p1 * mod + (p1 * (1 - mod))
  else:
    mod = abs(mod)
    mod = -p1 * mod + (p1 * (1 - mod))
  rel_accels3.append(mod)

# rel_accels3 = []
# for v_rel in v_rels:
#   p1 = (v_rel - v_rels[0]) / time
#   diff = False
#
#   if v_lead_change == 0 or v_ego_change == 0:
#     _print('equal/zero')
#     lead_factor = 1 if v_ego_change == 0 else 0
#
#   elif (v_ego_change < 0) != (v_lead_change < 0) and v_lead_change != 0 and v_ego_change != 0:
#     _print('here1')
#     # lead_factor = v_lead_change / abs(v_ego_change - v_lead_change)
#     lead_factor = v_lead_change / (v_lead_change - v_ego_change)
#     v_lead_change / (v_lead_change - v_ego_change)
#     diff = True
#
#   elif v_ego_change > 0 and v_lead_change > 0:  # make this and below the same
#     _print('here2')
#     lead_factor = v_lead_change / (v_ego_change + v_lead_change)
#     if v_ego_change > v_lead_change:
#       lead_factor = -lead_factor
#
#   elif v_ego_change < 0 and v_lead_change < 0:
#     lead_factor = v_lead_change / (v_ego_change + v_lead_change)
#     if v_ego_change > v_lead_change:
#       lead_factor = -lead_factor
#
#   else:
#     raise Exception('Uncovered case!')
#
#   if not diff:
#     lead_factor *= 1.25
#     if lead_factor > 0:
#       rel_vel_out = (p1 * abs(lead_factor)) + (-p1 * (1 - abs(lead_factor)))
#     else:
#       rel_vel_out = (-p1 * abs(lead_factor)) + (p1 * (1 - abs(lead_factor)))
#     rel_accels3.append(rel_vel_out)
#   else:
#     if v_ego_change < 0 and v_lead_change > 0:
#       rel_accels3.append(math.copysign(p1, v_lead_change - v_ego_change) * lead_factor)
#     else:
#       rel_accels3.append(math.copysign(-p1, v_lead_change - v_ego_change) * lead_factor)


# rel_vel_out = math.copysign(p1, v_lead_change - v_ego_change) * lead_factor
# print('distance m.1: {}'.format(np.interp(rel_vel_out, x, y)))

# print('distance m.2: {}'.format(np.interp(rel_vel_out, x, y)))


# print('lead_factor: {}'.format(lead_factor))

calc_TRs = [np.interp(accel, x, y) + TR for accel in rel_accels]
calc_TRs2 = [np.interp(accel, x, y) + TR for accel in rel_accels2]
calc_TRs3 = [np.interp(accel, x, y) + TR for accel in rel_accels3]

plt.figure(1)
plt.plot(np.linspace(0, 1, res), calc_TRs, label='TRs')
plt.plot(np.linspace(0, 1, res), calc_TRs2, label='TRs2')
plt.plot(np.linspace(0, 1, res), calc_TRs3, label='new TRs')
plt.legend()

plt.figure(2)
plt.plot(np.linspace(0, res, res), v_egos, label='v_ego')
plt.plot(np.linspace(0, res, res), v_leads, label='v_lead')
plt.legend()
plt.show()
