from pathlib import Path
import json
import numpy as np
from sdpo_result_feedback.env_adapter import PandaPushOutcomePilot,InvalidScene
from sdpo_result_feedback.scenes import create_scene_splits
from sdpo_result_feedback.actions import ACTION_LABELS
root=Path(__file__).resolve().parents[1]
scenes=create_scene_splits(root/'artifacts/scenes.json')['dev'][:20]
e=PandaPushOutcomePilot(max_decisions=32,seed=11)
rows=[]; rejected=[]
for scene in scenes:
 try: e.reset('phase_a_'+scene.scene_id,scene)
 except InvalidScene as x: rejected.append({'scene':scene.scene_id,'reason':str(x)}); continue
 snap=e.snapshot(); base=e._distance(); before=e._observation(); actions=[]
 for label in ACTION_LABELS:
  e.restore(snap)
  _,t=e.step(label)
  actions.append({'action':label,'distance_after_m':t.distance_after_m,'progress_m':t.distance_before_m-t.distance_after_m,'contact':t.result.contact,'object_motion':t.result.object_motion,'safety_failure':t.safety_failure})
 e.restore(snap); e.remove_snapshot(snap)
 rows.append({'scene_id':scene.scene_id,'distance_before_m':base,'actions':actions,'reset_audit':e.reset_audit})
(root/'artifacts/calibration/phase_a_probe.json').write_text(json.dumps({'n':len(rows),'rejected':rejected,'rows':rows},indent=2))
print(json.dumps({'scenes':len(rows),'rejected':len(rejected),'contact_rate':float(np.mean([a['contact'] for r in rows for a in r['actions']])), 'motion_rate':float(np.mean([a['object_motion']=='detected' for r in rows for a in r['actions']])),'mean_best_progress':float(np.mean([max(a['progress_m'] for a in r['actions']) for r in rows]))},indent=2))
e.close()
