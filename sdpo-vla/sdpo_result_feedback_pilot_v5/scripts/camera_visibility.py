import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from sdpo_result_feedback.env_adapter import PandaPushOutcomePilot,InvalidScene
from sdpo_result_feedback.scenes import propose_scene
root=Path(__file__).resolve().parents[1]
e=PandaPushOutcomePilot()
p=e.physics
projection=p.computeProjectionMatrixFOV(60,1,.1,100)
settings=[(yaw,pitch) for yaw in [0,180,225,270,315] for pitch in [-30,-45,-60]]
records={str(s):[] for s in settings}
for i in range(32):
 try: e.reset(scene=propose_scene(f'camera_calibration_{i:03d}'))
 except InvalidScene: continue
 for yaw,pitch in settings:
  view=p.computeViewMatrixFromYawPitchRoll([0,0,.02],.8,yaw,pitch,0,2)
  raw=p.getCameraImage(448,448,viewMatrix=view,projectionMatrix=projection,renderer=p.ER_TINY_RENDERER)
  seg=np.asarray(raw[4]).reshape(448,448)
  obj=int((seg==e.object_id).sum()); goal=int((seg==e.base.sim._bodies_idx['target']).sum())
  records[str((yaw,pitch))].append({'scene':i,'cube_pixels':obj,'target_pixels':goal})
summary=[]
for setting,rows in records.items():
 summary.append({'setting':setting,'both_visible':sum(r['cube_pixels']>=40 and r['target_pixels']>=40 for r in rows),'n':len(rows),'cube_mean':float(np.mean([r['cube_pixels'] for r in rows])),'min_cube':min(r['cube_pixels'] for r in rows)})
summary.sort(key=lambda r:(r['both_visible'],r['cube_mean']),reverse=True)
(root/'artifacts/calibration/camera_visibility.json').write_text(json.dumps({'summary':summary,'records':records},indent=2))
print(json.dumps(summary,indent=2))
e.close()
