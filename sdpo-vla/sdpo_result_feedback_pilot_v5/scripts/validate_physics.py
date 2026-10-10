"""CPU-only control and replay calibration; no labels are passed to a policy."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw
from sdpo_result_feedback.env_adapter import PandaPushOutcomePilot,InvalidScene,CAMERA,EE_HEIGHT_M
from sdpo_result_feedback.scenes import create_scene_splits
from sdpo_result_feedback.actions import ACTION_LABELS
root=Path(__file__).resolve().parents[1]
scenes=create_scene_splits(root/'artifacts/scenes.json')['dev'][:20]
e=PandaPushOutcomePilot()
rows=[];rejections=[];images=[]
for scene in scenes:
 try: obs=e.reset('physics_'+scene.scene_id,scene)
 except InvalidScene as error:
  rejections.append({'scene_id':scene.scene_id,'reason':str(error)});continue
 if len(images)<8:images.append((scene.scene_id,Image.fromarray(obs['image'])))
 state=e.snapshot()
 try:
  for action in ACTION_LABELS:
   e.restore(state)
   image1,t1=e.step(action)
   e.restore(state)
   image2,t2=e.step(action)
   rows.append({'scene_id':scene.scene_id,'action':action,
    'ee_delta_m':t1.actual_ee_delta_m,'ee_replay_error_m':float(np.max(np.abs(np.array(t1.actual_ee_delta_m)-t2.actual_ee_delta_m))),
    'object_replay_error_m':float(np.max(np.abs(np.array(t1.measurements['object_after'])-t2.measurements['object_after']))),
    'distance_replay_error_m':abs(t1.distance_after_m-t2.distance_after_m),
    'feedback_equal':t1.result==t2.result,
    'image_mae':float(np.abs(image1['image'].astype(float)-image2['image']).mean()),
    'object_motion_m':t1.measurements['object_motion_xy_m'],'contact':t1.result.contact})
 finally:e.remove_snapshot(state)
e.close()
summary={'scenes':len(scenes)-len(rejections),'reset_rejections':len(rejections),
 'probe_interactions':len(rows)*2,'max_object_replay_error_m':max(r['object_replay_error_m'] for r in rows),
 'max_distance_replay_error_m':max(r['distance_replay_error_m'] for r in rows),
 'max_ee_replay_error_m':max(r['ee_replay_error_m'] for r in rows),
 'max_image_mae':max(r['image_mae'] for r in rows),'all_feedback_equal':all(r['feedback_equal'] for r in rows),
 'max_hold_object_drift_m':max(r['object_motion_m'] for r in rows if r['action']=='E'),
 'camera':CAMERA,'ee_height_m':EE_HEIGHT_M}
for action in ACTION_LABELS:
 summary[f'{action}_mean_ee_delta_m']=np.mean([r['ee_delta_m'] for r in rows if r['action']==action],axis=0).tolist()
(root/'artifacts/calibration/physics_validation.json').write_text(json.dumps({'summary':summary,'rows':rows,'rejections':rejections},indent=2))
canvas=Image.new('RGB',(448*4,472*2),'white');draw=ImageDraw.Draw(canvas)
for i,(name,image) in enumerate(images):
 x,y=(i%4)*448,(i//4)*472
 draw.text((x+6,y+6),name,fill='black');canvas.paste(image,(x,y+24))
canvas.save(root/'artifacts/calibration/final_camera_contact_sheet.jpg')
print(json.dumps(summary,indent=2))
assert summary['max_object_replay_error_m']<.0002
assert summary['all_feedback_equal']
assert summary['max_hold_object_drift_m']<.002
