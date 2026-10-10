from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
import torch
from .env_adapter import InvalidScene, PandaPushOutcomePilot, Transition
from .feedback import causal_feedback, neutral_feedback, select_feedback
from .losses import reverse_kl_loss, symmetric_js_loss
from .scenes import Scene


@dataclass(frozen=True)
class TrainingConfig:
    transitions_per_update: int = 256
    microbatch_size: int = 4
    parallel_envs: int = 8
    max_decisions: int = 32
    seed: int = 11


def observation_transition(env, observation):
    return Transition(env.episode_id,env.decision_index,observation['image'],observation['instruction'],
        observation['proprio'],'E',None,[0.,0.,0.],0.,0.,False,False,env.scene.scene_id)


class RolloutCollector:
    """Persistent batched policy rollout; keep episodes across optimizer updates."""
    def __init__(self, config: TrainingConfig, scenes: list[Scene]):
        if not scenes or any(s.split!='train' for s in scenes):
            raise ValueError('Training requires explicitly train-only scenes')
        self.config,self.scenes=config,scenes
        self.rng=np.random.default_rng(config.seed)
        self.envs=[PandaPushOutcomePilot(config.max_decisions,config.seed) for _ in range(config.parallel_envs)]
        self.observations=[None]*len(self.envs)
        self.episode_counter=0
        self.resets=[]
        self.completed=[]
        for i in range(len(self.envs)):
            self._reset(i)

    def _reset(self,index):
        env=self.envs[index]
        for _ in range(100):
            scene=self.scenes[int(self.rng.integers(len(self.scenes)))]
            episode_id=f'train_s{self.config.seed}_{self.episode_counter:08d}'
            self.episode_counter+=1
            try:
                self.observations[index]=env.reset(episode_id,scene)
                self.resets.append({'episode_id':episode_id,**env.reset_audit,'accepted':True})
                return
            except InvalidScene as e:
                self.resets.append({'episode_id':episode_id,'scene_id':scene.scene_id,'accepted':False,'reason':str(e)})
        raise RuntimeError('Too many rejected reset proposals')

    def collect(self,policy):
        transitions=[]
        while len(transitions)<self.config.transitions_per_update:
            count=min(len(self.envs),self.config.transitions_per_update-len(transitions))
            inputs=[observation_transition(self.envs[i],self.observations[i]) for i in range(count)]
            actions=[]
            for start in range(0,count,self.config.microbatch_size):
                actions.extend(policy.sample_batch(inputs[start:start+self.config.microbatch_size]))
            for i,action in enumerate(actions):
                obs,t=self.envs[i].step(action)
                transitions.append(t)
                self.observations[i]=obs
                if t.terminated or t.truncated:
                    self.completed.append({'episode_id':t.episode_id,'success':t.result.success,'length':t.decision_index+1,
                                           'safety_failure':t.safety_failure})
                    self._reset(i)
        return transitions

    def close(self):
        for env in self.envs:
            env.close()


class GroupedRolloutCollector:
    """Sample several policy actions from one frozen initial robot state per group."""
    def __init__(self, config: TrainingConfig, scenes: list[Scene], candidates_per_scene=8):
        if not scenes or any(scene.split!='train' for scene in scenes):
            raise ValueError('Training requires explicitly train-only scenes')
        if candidates_per_scene<2 or config.transitions_per_update%candidates_per_scene:
            raise ValueError('Transitions per update must divide evenly into grouped candidates')
        self.config,self.scenes,self.candidates_per_scene=config,scenes,candidates_per_scene
        self.rng=np.random.default_rng(config.seed)
        self.env=PandaPushOutcomePilot(config.max_decisions,config.seed)
        self.group_counter=0
        self.resets=[]
        self.completed=[]

    def collect(self, policy):
        transitions=[]
        groups=self.config.transitions_per_update//self.candidates_per_scene
        for _ in range(groups):
            scene=self.scenes[int(self.rng.integers(len(self.scenes)))]
            group_id=f'train_g{self.config.seed}_{self.group_counter:06d}'
            self.group_counter+=1
            try:
                observation=self.env.reset(group_id,scene)
            except InvalidScene as error:
                self.resets.append({'episode_id':group_id,'scene_id':scene.scene_id,'accepted':False,'reason':str(error)})
                continue
            self.resets.append({'episode_id':group_id,**self.env.reset_audit,'accepted':True})
            initial=observation_transition(self.env,observation)
            actions=policy.sample_batch([initial]*self.candidates_per_scene)
            snapshot=self.env.snapshot()
            try:
                for candidate_index,action in enumerate(actions):
                    episode_id=f'{group_id}_c{candidate_index:02d}'
                    observation=self.env.restore(snapshot)
                    self.env.episode_id=episode_id
                    self.resets.append({'episode_id':episode_id,'scene_id':scene.scene_id,'accepted':True,
                                        'restored_from_snapshot':True,'rollout_group_id':group_id,
                                        'rollout_candidate_index':candidate_index})
                    _,transition=self.env.step(action)
                    transition.rollout_group_id=group_id
                    transition.rollout_candidate_index=candidate_index
                    transitions.append(transition)
                    self.completed.append({'episode_id':episode_id,'success':transition.result.success,
                                           'local_progress_witness':transition.result.goal_progress=='closer',
                                           'length':transition.decision_index+1,'safety_failure':transition.safety_failure,
                                           'rollout_group_id':group_id})
            finally:
                self.env.remove_snapshot(snapshot)
        if len(transitions)!=self.config.transitions_per_update:
            raise RuntimeError('Grouped collection did not produce the requested transition count')
        return transitions

    def close(self):
        self.env.close()


def build_group_teacher_context(transitions, condition, seed=11):
    """Build SDPO teacher context from same-state self-generated candidate rollouts."""
    allowed={'solution_only','solution_plus_causal','solution_plus_causal_shuffled','solution_plus_binary'}
    if condition not in allowed:
        raise ValueError(f'Unsupported grouped SDPO condition: {condition}')
    group_members={}
    for index,transition in enumerate(transitions):
        if transition.rollout_group_id is None:
            raise ValueError('Grouped SDPO requires rollout_group_id on every transition')
        group_members.setdefault(transition.rollout_group_id,[]).append(index)
    witnesses={}
    for group_id,members in group_members.items():
        witnesses[group_id]=next((index for index in members
                                  if transitions[index].safety_failure is None
                                  and not transitions[index].result.success
                                  and transitions[index].result.goal_progress=='closer'),None)
    solution_actions=[]
    for index,transition in enumerate(transitions):
        witness=witnesses[transition.rollout_group_id]
        solution_actions.append(None if witness is None or witness==index
                                else transitions[witness].action_label)
    if condition=='solution_only':
        records=[neutral_feedback() for _ in transitions]
        include_outcome_feedback=False
        donors=[None]*len(transitions)
    elif condition=='solution_plus_causal':
        records=[causal_feedback(transition.action_label,transition.result) for transition in transitions]
        include_outcome_feedback=True
        donors=list(range(len(transitions)))
    elif condition=='solution_plus_binary':
        records,_,donors=select_feedback(transitions,'binary',seed)
        include_outcome_feedback=True
    else:
        records,_,donors=select_feedback(transitions,'causal_shuffled',seed)
        include_outcome_feedback=True
    local_witness=np.asarray([witnesses[t.rollout_group_id] is not None
                              and witnesses[t.rollout_group_id]==index
                              for index,t in enumerate(transitions)],dtype=bool)
    usable_solution=np.asarray([action is not None for action in solution_actions],dtype=bool)
    safe_failed=np.asarray([t.safety_failure is None and not t.result.success for t in transitions],dtype=bool)
    valid=safe_failed & ~local_witness & (usable_solution | include_outcome_feedback)
    return {'condition':condition,'records':records,'solution_actions':solution_actions,
            'include_outcome_feedback':include_outcome_feedback,'valid_mask':valid,
            'donor_indices':donors,'group_count':len(group_members),'witness_count':int(local_witness.sum()),
            'groups_with_witness':sum(witness is not None for witness in witnesses.values()),
            'solution_context_count':int(usable_solution.sum())}


def sdpo_update(policy,transitions,feedback_mode='result',microbatch_size=4,seed=11,teacher_context=None,
                reflection_records=None):
    if not transitions or microbatch_size<1:
        raise ValueError('Need transitions and positive microbatch size')
    policy.model.eval()  # Disable dropout, not gradients.
    if teacher_context is None:
        records,valid,donors=select_feedback(transitions,feedback_mode,seed)
        if reflection_records is not None:
            if len(reflection_records)!=len(transitions):
                raise ValueError('Reflection records must align one-to-one with transitions')
            records=list(reflection_records)
        # Original SDPO rich-feedback uses environment feedback whenever available
        # and skips reprompting a sample that already succeeded. Every safe failed
        # robot attempt has a measured outcome, including no-contact outcomes.
        if feedback_mode!='no_feedback':
            valid=valid & np.asarray([not t.result.success for t in transitions],dtype=bool)
        solution_actions=None
        include_outcome_feedback=True
    else:
        records=teacher_context['records']
        valid=np.asarray(teacher_context['valid_mask'],dtype=bool)
        donors=teacher_context['donor_indices']
        solution_actions=teacher_context['solution_actions']
        include_outcome_feedback=teacher_context['include_outcome_feedback']
    valid_count=int(valid.sum())
    if feedback_mode=='no_feedback':
        return {'loss':0.,'valid':0,'executed':len(transitions),'gradient_norm':0.,
                'optimizer_steps':0,'seconds':0.,'donor_indices':donors,
                'valid_mask':[False]*len(transitions),'teacher_log_probs':[]}
    if not valid_count:
        raise RuntimeError('No valid transitions; refusing an empty optimizer step')
    start=time.perf_counter()
    teacher=torch.full((len(transitions),5),-float(np.log(5.)),dtype=torch.float32)
    # Freeze the teacher for this entire round before ANY optimizer step.
    with torch.no_grad():
        for i in range(0,len(transitions),microbatch_size):
            selected=np.flatnonzero(valid[i:i+microbatch_size])+i
            if not len(selected):
                continue
            selected_transitions=[transitions[index] for index in selected]
            teacher_fn=getattr(policy,'teacher_log_probs',None)
            teacher_kwargs={'feedback_records':[records[index] for index in selected]}
            if teacher_context is not None:
                teacher_kwargs.update({'solution_actions':[solution_actions[index] for index in selected],
                                       'include_outcome_feedback':(
                                           [include_outcome_feedback[index] for index in selected]
                                           if isinstance(include_outcome_feedback,list)
                                           else include_outcome_feedback)})
            if teacher_fn is None:
                lp=policy.log_probs(selected_transitions,feedback=True,grad=False,
                                    **teacher_kwargs)
            else:
                lp=teacher_fn(selected_transitions,feedback=True,
                              **teacher_kwargs)
            if not torch.isfinite(lp).all():
                raise FloatingPointError('Nonfinite teacher probabilities')
            teacher[selected]=lp.detach().cpu()
    policy.optimizer.zero_grad(set_to_none=True)
    total_loss=0.
    try:
        for i in range(0,len(transitions),microbatch_size):
            mask=torch.as_tensor(valid[i:i+microbatch_size],device=policy.device)
            if not mask.any():
                continue
            student=policy.log_probs(transitions[i:i+microbatch_size],feedback=False,grad=True)
            q=teacher[i:i+microbatch_size].to(policy.device)
            loss_name=getattr(getattr(policy,'config',None),'distillation_loss','reverse_kl')
            loss_function=symmetric_js_loss if loss_name=='js' else reverse_kl_loss
            loss=loss_function(student,q,mask)*int(mask.sum())/valid_count
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite student loss')
            loss.backward()
            total_loss+=float(loss.detach())
        norm=torch.nn.utils.clip_grad_norm_(list(policy.trainable_parameters()),1.,error_if_nonfinite=True)
        policy.optimizer.step()  # Exactly once, after accumulation over the full rollout batch.
        if hasattr(policy,'update_teacher_ema'):
            policy.update_teacher_ema()
    except Exception:
        policy.optimizer.zero_grad(set_to_none=True)
        raise
    policy.optimizer.zero_grad(set_to_none=True)
    student_probs=[]
    teacher_probs=teacher.exp()
    teacher_argmax=teacher_probs.argmax(-1).numpy()
    attempted=np.asarray(['ABCDE'.index(t.action_label) for t in transitions])
    positive=np.asarray([t.result.goal_progress=='closer' for t in transitions]) & valid
    negative=(~positive) & valid
    oracle=[]
    for transition in transitions:
        measurements=getattr(transition,'measurements',{})
        if 'object_before' not in measurements or 'goal_xy' not in measurements:
            oracle.append(-1)
            continue
        obj=np.asarray(measurements['object_before'][:2])
        goal=np.asarray(measurements['goal_xy'])
        delta=goal-obj
        if abs(delta[0])>=abs(delta[1]):
            oracle.append(0 if delta[0]>0 else 1)
        else:
            oracle.append(2 if delta[1]>0 else 3)
    oracle=np.asarray(oracle)
    oracle_valid=valid & (oracle>=0)
    for i in range(0,len(transitions),microbatch_size):
        with torch.no_grad():
            student_probs.append(policy.log_probs(transitions[i:i+microbatch_size],feedback=False,grad=False).exp().cpu())
    student_probs=torch.cat(student_probs)
    tv=.5*(student_probs-teacher_probs).abs().sum(-1)
    entropy=-(teacher_probs*teacher.clamp_min(-100.)).sum(-1)
    return {'loss':total_loss,'valid':valid_count,'executed':len(transitions),'gradient_norm':float(norm),
            'optimizer_steps':1,'seconds':time.perf_counter()-start,'donor_indices':donors,
            'valid_mask':valid.tolist(),'teacher_log_probs':teacher.tolist(),
            'mean_teacher_student_total_variation':float(tv[valid].mean()),
            'mean_teacher_entropy':float(entropy[valid].mean()),
            'teacher_argmax_counts':{str(i):int((teacher_argmax[valid]==i).sum()) for i in range(5)},
            'teacher_oracle_accuracy':float((teacher_argmax[oracle_valid]==oracle[oracle_valid]).mean()) if oracle_valid.any() else None,
            'teacher_repeat_positive_accuracy':float((teacher_argmax[positive]==attempted[positive]).mean()) if positive.any() else None,
            'teacher_reject_negative_accuracy':float((teacher_argmax[negative]!=attempted[negative]).mean()) if negative.any() else None}
