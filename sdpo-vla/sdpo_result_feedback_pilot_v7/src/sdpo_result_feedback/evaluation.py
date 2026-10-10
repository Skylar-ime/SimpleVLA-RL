"""No-feedback deployment evaluation and diagnostic-only counterfactual probes."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import time
import numpy as np
from PIL import Image
from .actions import ACTION_LABELS
from .env_adapter import PandaPushOutcomePilot,InvalidScene
from .feedback import (directional_feedback, outcome_verdict, select_feedback,
                        wrong_directional_feedback)
from .trainer import observation_transition


def json_write(path, data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def evaluate(policy,scenes,output=None,microbatch_size=4,max_decisions=32):
    start=time.perf_counter()
    rows=[]; interactions=0
    envs=[PandaPushOutcomePilot(max_decisions=max_decisions) for _ in range(microbatch_size)]
    try:
        for offset in range(0,len(scenes),microbatch_size):
            batch=scenes[offset:offset+microbatch_size]
            active=[]; observations={}; trajectories={}; initial_distances={}
            for i,scene in enumerate(batch):
                try:
                    observations[i]=envs[i].reset(f'eval_{scene.scene_id}',scene)
                    active.append(i);trajectories[i]=[];initial_distances[i]=envs[i]._distance()
                except InvalidScene as exc:
                    rows.append({'scene_id':scene.scene_id,'success':False,'length':0,
                                 'status':'invalid_reset','reason':str(exc)})
            while active:
                inputs=[observation_transition(envs[i],observations[i]) for i in active]
                labels=policy.sample_batch(inputs,greedy=True)
                next_active=[]
                for i,label in zip(active,labels):
                    observations[i],t=envs[i].step(label)
                    interactions+=1
                    trajectories[i].append(t.as_json_record())
                    if t.terminated or t.truncated:
                        rows.append({'scene_id':t.scene_id,'success':t.result.success,'length':t.decision_index+1,
                            'status':'safety_failure' if t.safety_failure else 'success' if t.result.success else 'timeout',
                            'initial_distance_m':initial_distances[i],'final_distance_m':t.distance_after_m,
                            'progress_m':initial_distances[i]-t.distance_after_m,'trajectory':trajectories[i]})
                    else:
                        next_active.append(i)
                active=next_active
    finally:
        for env in envs: env.close()
    valid=[r for r in rows if r['status']!='invalid_reset']
    successes=sum(r['success'] for r in valid)
    summary={'scenes_requested':len(scenes),'scenes_valid':len(valid),'invalid_resets':len(rows)-len(valid),
        'successes':successes,'success_rate':successes/len(valid) if valid else None,
        'conservative_success_rate_all_scenes':successes/len(scenes) if scenes else None,
        'mean_progress_m':float(np.mean([r['progress_m'] for r in valid])) if valid else None,
        'safety_failures':sum(r['status']=='safety_failure' for r in rows),
        'interactions':interactions,'seconds':time.perf_counter()-start,
        'action_counts':dict(Counter(t['action_label'] for r in valid for t in r['trajectory']))}
    report={'summary':summary,'episodes':rows}
    if output: json_write(output,report)
    return report


def feedback_probe(policy,scenes,output_dir,microbatch_size=4,seed=11):
    """Initial-state paired diagnostic. Never use counterfactual outcomes as labels."""
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    env=PandaPushOutcomePilot(seed=seed)
    transitions=[]; counterfactual=[]; rejected=[]; replay=[]
    start=time.perf_counter()
    try:
        for index,scene in enumerate(scenes):
            try: obs=env.reset(f'probe_{scene.scene_id}',scene)
            except InvalidScene as exc:
                rejected.append({'scene_id':scene.scene_id,'reason':str(exc)})
                continue
            snap=env.snapshot()
            try:
                # Canonicalize the restored state before every branch, including the
                # executed attempt. Initial restore and replay must use the same path.
                obs=env.restore(snap)
                original=observation_transition(env,obs)
                action=policy.sample_batch([original])[0]
                after,t=env.step(action)
                env.restore(snap);again,t2=env.step(action)
                replay.append({'scene_id':scene.scene_id,
                    'distance_error_m':abs(t.distance_after_m-t2.distance_after_m),
                    'pixel_mean_absolute_error':float(np.abs(after['image'].astype(float)-again['image']).mean()),
                    'feedback_equal':t.result==t2.result})
                if abs(t.distance_after_m-t2.distance_after_m)>.0002 or t.result!=t2.result:
                    raise RuntimeError(f'Counterfactual replay failed tolerance: {scene.scene_id}')
                values=[];contacts=[];safety=[]
                for label in ACTION_LABELS:
                    env.restore(snap)
                    _,cf=env.step(label)
                    values.append(cf.distance_before_m-cf.distance_after_m)
                    contacts.append(cf.result.contact)
                    safety.append(cf.safety_failure is not None)
                transitions.append(t)
                counterfactual.append({'scene_id':scene.scene_id,'progress_m':values,'contact':contacts,'unsafe':safety})
                if index<8: Image.fromarray(t.image).save(out/f'{scene.scene_id}.png')
            finally:
                env.remove_snapshot(snap)
    finally:
        env.close()
    if not transitions: raise RuntimeError('No valid probe transitions')
    def distributions(feedback=False,records=None,teacher=False):
        chunks=[]
        for i in range(0,len(transitions),microbatch_size):
            kwargs = {
                'feedback': feedback,
                'feedback_records': None if records is None else records[i:i+microbatch_size],
            }
            scorer = policy.teacher_log_probs if teacher else policy.log_probs
            if not teacher:
                kwargs['grad'] = False
            lp=scorer(transitions[i:i+microbatch_size],**kwargs)
            chunks.append(lp.exp().cpu().numpy())
        return np.concatenate(chunks)
    p=distributions(); qs={}; donor_map={}; matched={}
    for mode in ['result','result_interpreted','causal','causal_shuffled','binary','shuffled','hard_shuffled','neutral','no_feedback']:
        records,valid,donors=select_feedback(transitions,mode,seed)
        qs[mode]=distributions(mode!='no_feedback',records,teacher=mode!='no_feedback')
        donor_map[mode]=donors
        matched[mode]={'matched':sum(d is not None for d in donors),
            'changed':sum((outcome_verdict(record)!=outcome_verdict(t.result))
                          if mode=='causal_shuffled' else record!=t.result
                          for record,t in zip(records,transitions))}
    for mode, builder in [('directional', directional_feedback),
                          ('wrong_directional', wrong_directional_feedback)]:
        records = [builder(t.action_label, t) for t in transitions]
        qs[mode] = distributions(True, records, teacher=True)
        donor_map[mode] = list(range(len(transitions)))
        matched[mode] = {'matched': len(transitions),
                         'changed': sum(record != t.result
                                        for record, t in zip(records, transitions))}
    values=np.array([r['progress_m'] for r in counterfactual])
    contacts=np.array([r['contact'] for r in counterfactual],dtype=float)
    base=(p*values).sum(-1)
    rng=np.random.default_rng(seed)
    summaries={}
    for mode,q in qs.items():
        difference=(q*values).sum(-1)-base
        bootstrap=np.mean(rng.choice(difference,size=(2000,len(difference)),replace=True),axis=1)
        summaries[mode]={'expected_progress_difference_m':float(difference.mean()),
            'bootstrap95_progress_difference_m':np.quantile(bootstrap,[.025,.975]).tolist(),
            'mean_total_variation':float(.5*np.abs(q-p).sum(-1).mean()),
            'expected_contact_difference':float(((q-p)*contacts).sum(-1).mean()),
            'matching':matched[mode]}
    informative=np.asarray([t.result.goal_progress!='unchanged' or t.result.contact for t in transitions])
    attempted=np.asarray([ACTION_LABELS.index(t.action_label) for t in transitions])
    oracle=[]
    for transition in transitions:
        obj=np.asarray(transition.measurements['object_before'][:2])
        goal=np.asarray(transition.measurements['goal_xy'])
        delta=goal-obj
        if abs(delta[0])>=abs(delta[1]):
            oracle.append(0 if delta[0]>0 else 1)
        else:
            oracle.append(2 if delta[1]>0 else 3)
    oracle=np.asarray(oracle)
    for mode,q in qs.items():
        argmax=q.argmax(-1)
        mask=informative & np.asarray([not t.result.success for t in transitions])
        summaries[mode]['teacher_repeat_previous_accuracy']=float((argmax[mask]==attempted[mask]).mean()) if mask.any() else None
        summaries[mode]['teacher_oracle_accuracy']=float((argmax==oracle).mean())
        summaries[mode]['teacher_argmax_counts']={label:int((argmax[mask]==i).sum()) for i,label in enumerate(ACTION_LABELS)}
    n=len(transitions)
    report={'summary':{'scenes':n,'rejected':len(rejected),'seconds':time.perf_counter()-start,
        'executed_interactions':7*n,'attempt_interactions':n,'replay_interactions':n,'counterfactual_interactions':5*n,
        'attempt_contact_count':sum(t.result.contact for t in transitions),
        'counterfactual_contact_count':int(contacts.sum()),
        'informative_progress_scenes':int((np.ptp(values,axis=1)>.002).sum()),
        'groups':summaries,'note':'Initial-state local diagnosis only; no training or action supervision.'},
        'records':[t.as_json_record() for t in transitions],'student_probs':p.tolist(),
        'teacher_probs':{k:v.tolist() for k,v in qs.items()},'counterfactual':counterfactual,
        'donors':donor_map,'replay':replay,'rejected':rejected}
    json_write(out/'report.json',report)
    return report
