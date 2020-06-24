import cereal.messaging as messaging
from cereal.messaging import PubMaster

pm = PubMaster(['laneSpeed'])

input('press enter')
msg = messaging.new_message('laneSpeed')
msg.laneSpeed.leftAvgSpeed = 42.4
msg.laneSpeed.middleAvgSpeed = 38.4
msg.laneSpeed.rightAvgSpeed = 54.4
msg.laneSpeed.status = 'right'
msg.laneSpeed.new = False
pm.send('laneSpeed', msg)

input('press enter')

msg = messaging.new_message('laneSpeed')
msg.laneSpeed.status = 'right'
msg.laneSpeed.new = False
pm.send('laneSpeed', msg)

input('press enter')

msg = messaging.new_message('laneSpeed')
msg.laneSpeed.leftAvgSpeed = 10.
msg.laneSpeed.middleAvgSpeed = 20.
msg.laneSpeed.rightAvgSpeed = 30.
msg.laneSpeed.new = False
pm.send('laneSpeed', msg)
