"""Hardware and optimizer contract smoke test; not an effectiveness experiment."""
from pathlib import Path
import json,time,hashlib
import torch
from sdpo_result_feedback.policy import QwenActionPolicy
from sdpo_result_feedback.trainer import observation_transition,sdpo_update
from sdpo_result_feedback.env_adapter import PandaPushOutcomePilot

root=Path(__file__).resolve().parents[1]
out=root/'artifacts/model_smoke'
out.mkdir(parents=True,exist_ok=True)
start=time.perf_counter()
print('MODEL_LOAD_START',flush=True)
policy=QwenActionPolicy()
print('MODEL_LOADED',time.perf_counter()-start,flush=True)
(out/'policy_audit.json').write_text(json.dumps(policy.audit,indent=2))
env=PandaPushOutcomePilot(); records=[]
for i,label in enumerate(['A','B','C','D']):
 obs=env.reset(f'model_smoke_{i}')
 inp=observation_transition(env,obs)
 p=policy.log_probs([inp]).exp().cpu().tolist()[0]
 _,t=env.step(label); records.append(t)
 print('ACTION_PROBS',i,p,flush=True)
# Measure BF16 batched-vs-single numerical drift rather than assuming invariance.
p_single=policy.log_probs([records[0]],True).cpu()
p_batch=policy.log_probs(records,True).cpu()
padding_error=float((p_single-p_batch[:1]).abs().max())
probability_error=float((p_single.exp()-p_batch[:1].exp()).abs().max())
# Also test a duplicated input: same pixels/text, no heterogeneous padding.
p_duplicate=policy.log_probs([records[0]]*4,True).cpu()
duplicate_error=float((p_single.exp()-p_duplicate[:1].exp()).abs().max())
trainable={n:p.detach().cpu().clone() for n,p in policy.model.named_parameters() if p.requires_grad}
print('UPDATE_START',flush=True)
update=sdpo_update(policy,records,microbatch_size=2)
changed=sum(not torch.equal(before,dict(policy.model.named_parameters())[n].detach().cpu()) for n,before in trainable.items())
report={'status':'passed' if changed>0 and update['gradient_norm']>0 and probability_error<.02 else 'failed',
        'elapsed_seconds':time.perf_counter()-start,'padding_logprob_max_error':padding_error,'padding_probability_max_error':probability_error,
        'duplicate_batch_probability_max_error':duplicate_error,
        'changed_trainable_tensors':changed,'gpu_name':torch.cuda.get_device_name(),
        'gpu_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'gpu_peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'update':update,
        'note':'Four-transition engineering smoke only; weights discarded after exit, no reported learning result.'}
(out/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='update'},indent=2),flush=True)
print('SMOKE_COMPLETE',flush=True)
env.close()
if report['status']!='passed': raise SystemExit(1)
