import cereal.messaging as messaging


sm = messaging.SubMaster(['modelV2'])

while 1:
  sm.update(0)

  if len(sm['modelV2'].laneLines) != 4:
    continue

  print('left lane line:')
  print(list(sm['modelV2'].laneLines[1].y))

  print('right lane line:')
  print(list(sm['modelV2'].laneLines[2].y))

  input()

