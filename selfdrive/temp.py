import sys
import time
from cereal.messaging import SubMaster

sm = SubMaster(['logMessage'])

last = ''
while 1:
  while sm['logMessage']['msg'] == last:
    sm.update(0)
  last = sm['logMessage']['msg']
  print(last)
# print("PRINT")