"""Post-pilot visual sensitivity and adapter drift audit on dev only; no updates."""
from pathlib import Path
from dataclasses import replace
from collections import Counter
import argparse,json,time
import numpy as np
import torch
from PIL import Image,ImageDraw
from sdpo_result_feedback.policy import QwenActionPolicy
from sdpo_result_feedback.env_adapter import PandaPushOutcomePilot,InvalidScene
from sdpo_result_feedback.scenes import create_directional_long_scene_splits,create_directional_scene_splits,create_mechanism_scene_splits,create_scene_splits
from sdpo_result_feedback.trainer import observation_transition
from sdpo_result_feedback.evaluation import json_write


def color_mask(image,kind):
    a=image.astype(float)
    if kind=='cube':
        region=(a[:,:,1]>60)&(a[:,:,1]>a[:,:,0]*1.7)&(a[:,:,1]>a[:,:,2]*1.7)
    else:
        region=(a[:,:,2]>60)&(a[:,:,2]>a[:,:,0]*1.7)&(a[:,:,2]>a[:,:,1]*1.7)
        region[340:,:130]=False  # Exclude the fixed +y legend, not scene geometry.
    result=image.copy()
    y,x=np.where(region)
    if not len(x):return result,0
    x0,x1=max(0,x.min()-4),min(448,x.max()+5)
    y0,y1=max(0,y.min()-4),min(448,y.max()+5)
    result[y0:y1,x0:x1]=[190,190,190]
    return result,int(len(x))


def distribution(policy,inputs,batch=4):
    return np.concatenate([policy.log_probs(inputs[i:i+batch],grad=False).exp().cpu().numpy()
                           for i in range(0,len(inputs),batch)])


def compare(original,modified):
    return {'mean_total_variation':float(np.abs(original-modified).sum(-1).mean()/2),
        'max_total_variation':float(np.max(np.abs(original-modified).sum(-1)/2)),
        'argmax_changed_count':int(np.count_nonzero(original.argmax(-1)!=modified.argmax(-1))),
        'kl_original_to_modified':float(np.mean(np.sum(original*np.log(np.clip(original,1e-30,1)/np.clip(modified,1e-30,1)),axis=-1)))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];run=args.directory.resolve();out=run/'grounding_audit'
    if out.exists():raise SystemExit(f'Refusing to overwrite {out}')
    groups=['frozen']+[name for name in ['result','result_interpreted','binary','shuffled','hard_shuffled','neutral','no_feedback']
                       if (run/name/'completed.json').exists()]
    for g in groups:
        if not (run/g/'completed.json').exists():raise SystemExit(f'Incomplete run {g}')
    out.mkdir();start=time.perf_counter()
    run_config=json.loads((run/groups[-1]/'run_config.json').read_text())
    scene_profile=run_config['arguments'].get('scene_profile','standard')
    scenes=((create_mechanism_scene_splits(root/'artifacts/mechanism_scenes.json')
             if scene_profile=='mechanism_near_contact'
             else create_directional_long_scene_splits(root/'artifacts/directional_long_scenes_v1.json')
             if scene_profile=='directional_long'
             else create_directional_scene_splits(root/'artifacts/directional_scenes_v2.json')
             if scene_profile=='directional_push'
             else create_scene_splits(root/'artifacts/scenes.json'))['dev'][:32])
    env=PandaPushOutcomePilot();inputs=[];rejected=[]
    try:
        for scene in scenes:
            try:obs=env.reset(f'grounding_{scene.scene_id}',scene)
            except InvalidScene as error:rejected.append(str(error));continue
            inputs.append(observation_transition(env,obs))
    finally:env.close()
    variants={'original':inputs};coverage={}
    for kind in ['cube','target']:
        masked=[color_mask(t.image,kind) for t in inputs]
        variants[f'mask_{kind}']=[replace(t,image=image) for t,(image,n) in zip(inputs,masked)]
        coverage[kind]=[n for image,n in masked]
    variants['shuffled_image']=[replace(t,image=inputs[(i+1)%len(inputs)].image.copy()) for i,t in enumerate(inputs)]
    gray=[]
    for t in inputs:
        image=np.full_like(t.image,190)
        image[340:,:130]=t.image[340:,:130]  # Keep fixed world-axis legend unchanged.
        gray.append(replace(t,image=image))
    variants['gray_scene']=gray
    sheet=Image.new('RGB',(448*5,476),'white');draw=ImageDraw.Draw(sheet)
    for i,(name,rows) in enumerate(variants.items()):
        draw.text((448*i+8,6),name,fill='black');sheet.paste(Image.fromarray(rows[0].image),(448*i,28))
    sheet.save(out/'input_variants.jpg')
    policy=QwenActionPolicy();report={'scope':'32 fixed dev initial states; inference only; no test opening',
        'rejected':rejected,'mask_visible_pixels':coverage,'groups':{}}
    frozen=None
    for group in groups:
        if group!='frozen':
            group_config=json.loads((run/group/'run_config.json').read_text())
            final_update=int(group_config['arguments']['updates'])
            checkpoint=run/group/f'update_{final_update:03d}'/'checkpoint'
            policy.model.load_adapter(str(checkpoint),adapter_name=group,is_trainable=False)
            policy.model.set_adapter(group)
            policy.model.eval()
        distributions={name:distribution(policy,rows) for name,rows in variants.items()}
        base=distributions['original']
        repeated=distribution(policy,inputs)
        if frozen is None:frozen=base.copy()
        summary={'sample_count':len(inputs),'mean_action_probabilities':base.mean(0).tolist(),
            'greedy_action_counts':dict(Counter('ABCDE'[i] for i in base.argmax(-1))),
            'mean_entropy_nats':float(np.mean(-np.sum(base*np.log(np.clip(base,1e-30,1)),axis=1))),
            'drift_from_frozen':compare(frozen,base),'repeat_numerical_noise':compare(base,repeated),
            'image_interventions':{name:compare(base,p) for name,p in distributions.items() if name!='original'}}
        # Fixed, bounded batch records real sequence sizes and time without updates.
        timings={}
        for feedback in [False,True]:
            ts=inputs[:4]
            if feedback:
                from sdpo_result_feedback.feedback import FeedbackRecord
                ts=[replace(t,result=FeedbackRecord(False,'none','unchanged',False)) for t in ts]
            encoded=policy._inputs(ts,feedback)
            timings['teacher' if feedback else 'student']={'sequence_lengths':encoded['attention_mask'].sum(-1).cpu().tolist(),
                'padded_length':int(encoded['input_ids'].shape[1]),'image_grid_thw':encoded['image_grid_thw'].cpu().tolist()}
        summary['token_shapes']=timings
        # Audit whether the constrained A-E policy conditions on plausible answer
        # tokens or on a tiny subset of the model's unrestricted next-token mass.
        with torch.no_grad():
            encoded=policy._inputs(inputs[:1],False)
            base_model=policy.model.get_base_model()
            body=base_model.model(**encoded,use_cache=False,return_dict=True)
            features=body.last_hidden_state[:,-1,:].float()
            all_logits=torch.nn.functional.linear(features,base_model.lm_head.weight.float())
            all_probs=all_logits.softmax(-1)[0]
            values,indices=torch.topk(all_probs,10)
            summary['unrestricted_answer_probe']={
                'scene_id':inputs[0].scene_id,
                'action_token_probability_mass':float(all_probs[policy.action_token_ids].sum()),
                'top_tokens':[{'token_id':int(i),'decoded':policy.processor.tokenizer.decode([int(i)]),'probability':float(v)}
                              for i,v in zip(indices.cpu(),values.cpu())]}
            del body,features,all_logits,all_probs,encoded
        report['groups'][group]={'summary':summary,'probabilities':{k:v.tolist() for k,v in distributions.items()}}
        json_write(out/'report.json',report)
        print(json.dumps({'group':group,**summary}),flush=True)
    report['seconds']=time.perf_counter()-start
    report['limitations']=['Distribution sensitivity is not proof of correct visual grounding or successful control.',
        'Color masks can include shadows; mask coverage is reported. Occluded objects may have no mask.',
        'Adapter drift and sensitivity are measured on these dev states, not all deployment states.',
        'No training or stochastic action sampling occurred in this audit.']
    json_write(out/'report.json',report)
    print('GROUNDING_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':main()
