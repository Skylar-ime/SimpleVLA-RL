import json
from pathlib import Path
import gymnasium as gym
import numpy as np
import panda_gym

out=Path(__file__).resolve().parents[1]/'artifacts/calibration'
env=gym.make('PandaPush-v3',render_mode='rgb_array',renderer='Tiny',render_width=448,render_height=448, render_distance=.7,render_yaw=90,render_pitch=-65,render_target_position=np.array([0.,0.,.02]))
b=env.unwrapped
p=b.sim.physics_client
rid=b.sim._bodies_idx['panda']; oid=b.sim._bodies_idx['object']
records=[]
for height in [.025,.035,.045,.055,.065,.075]:
 for angle in [0.,np.pi/2,np.pi,3*np.pi/2]:
  env.reset(seed=1)
  b.task.goal=np.array([.1,0.,.02])
  b.sim.set_base_pose('target',b.task.goal,np.array([0.,0.,0.,1.]))
  b.sim.set_base_pose('object',np.array([0.,0.,.02]),np.array([0.,0.,0.,1.]))
  target=np.array([.07*np.cos(angle),.07*np.sin(angle),height])
  for _ in range(25):
   arm=b.robot.inverse_kinematics(11,target,np.array([1.,0.,0.,0.]))[:7]
   joints=np.r_[arm,0.,0.]
   b.robot.set_joint_angles(joints)
  b.robot.control_joints(joints)
  p.performCollisionDetection()
  initial=p.getClosestPoints(rid,oid,.1)
  initial_contact=[(c[3],c[8]) for c in initial if c[8]<0]
  for _ in range(10): b.sim.step()
  drift=np.linalg.norm(b.task.get_achieved_goal()[:2])
  ee_before=b.robot.get_ee_position().copy()
  contacts=[]
  for _ in range(4):
   b.robot.set_action(np.array([-.4*np.cos(angle),-.4*np.sin(angle),0.]))
   for _ in range(4):
    b.sim.step()
    contacts.extend([(c[3],c[8]) for c in p.getContactPoints(rid,oid)])
  records.append(dict(height=height,angle=float(angle),initial_contact=initial_contact,settle_drift=float(drift),ee_before=ee_before.tolist(),object_after=b.task.get_achieved_goal().tolist(),contacts=contacts[-8:]))
print(json.dumps(records,indent=2))
(out/'geometry_sweep.json').write_text(json.dumps(records,indent=2))
env.close()
