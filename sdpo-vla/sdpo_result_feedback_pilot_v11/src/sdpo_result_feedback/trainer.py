from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import time
import numpy as np
import torch
from .env_adapter import ChunkTransition, InvalidScene, PandaPushOutcomePilot, Transition
from .actions import ACTION_LABELS
from .feedback import (aggregate_feedback, causal_feedback, neutral_feedback,
                       counterfactual_advantage_feedback, no_op_directional_feedback,
                       select_feedback)
from .losses import apply_teacher_residual, reverse_kl_loss, symmetric_js_loss
from .scenes import Scene


@dataclass(frozen=True)
class TrainingConfig:
    transitions_per_update: int = 256
    microbatch_size: int = 4
    parallel_envs: int = 8
    max_decisions: int = 32
    seed: int = 11
    chunk_length: int = 2


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
                counterfactual = {}
                counterfactual_transitions = {}
                for label in ACTION_LABELS:
                    self.env.restore(snapshot, render_observation=False)
                    self.env.episode_id = f'{group_id}_cf_{label}'
                    _, counterfactual_transition = self.env.step(
                        label, render_observation=False, pre_action_observation=initial)
                    counterfactual_transitions[label] = counterfactual_transition
                    counterfactual[label] = {
                        'progress_m': float(counterfactual_transition.result.progress_m),
                        'contact': bool(counterfactual_transition.result.contact),
                        'safety_failure': counterfactual_transition.safety_failure is not None,
                    }
                counterfactual_progress = [counterfactual[label]['progress_m']
                                           for label in ACTION_LABELS]
                counterfactual_contact = [counterfactual[label]['contact']
                                          for label in ACTION_LABELS]
                counterfactual_failure = [counterfactual[label]['safety_failure']
                                          for label in ACTION_LABELS]
                for candidate_index,action in enumerate(actions):
                    episode_id=f'{group_id}_c{candidate_index:02d}'
                    transition=deepcopy(counterfactual_transitions[action])
                    transition.episode_id=episode_id
                    transition.action_label=action
                    transition.rollout_group_id=group_id
                    transition.rollout_candidate_index=candidate_index
                    self.resets.append({'episode_id':episode_id,'scene_id':scene.scene_id,'accepted':True,
                                        'restored_from_snapshot':True,'rollout_group_id':group_id,
                                        'rollout_candidate_index':candidate_index})
                    transition.measurements.update({
                        'counterfactual_progress_m': counterfactual_progress,
                        'counterfactual_contact': counterfactual_contact,
                        'counterfactual_safety_failure': counterfactual_failure,
                        'action_advantage_m': float(
                            counterfactual_progress[ACTION_LABELS.index(action)]
                            - counterfactual_progress[ACTION_LABELS.index('E')]),
                    })
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


def _aggregate_chunk_feedback(step_results):
    return aggregate_feedback(step_results)


class ChunkRolloutCollector:
    """Sample fixed-length autoregressive chunks from one frozen initial state."""
    def __init__(self, config: TrainingConfig, scenes: list[Scene],
                 candidates_per_scene=8):
        if not scenes or any(scene.split != 'train' for scene in scenes):
            raise ValueError('Training requires explicitly train-only scenes')
        if config.chunk_length < 2:
            raise ValueError('Chunk length must be at least two')
        if candidates_per_scene < 2 or config.transitions_per_update % candidates_per_scene:
            raise ValueError('Transitions per update must divide evenly into grouped candidates')
        self.config, self.scenes = config, scenes
        self.candidates_per_scene = candidates_per_scene
        self.rng = np.random.default_rng(config.seed)
        self.env = PandaPushOutcomePilot(config.chunk_length, config.seed)
        self.group_counter = 0
        self.resets = []
        self.completed = []

    def collect(self, policy):
        chunks = []
        groups = self.config.transitions_per_update // self.candidates_per_scene
        for _ in range(groups):
            scene = self.scenes[int(self.rng.integers(len(self.scenes)))]
            group_id = f'train_chunk_g{self.config.seed}_{self.group_counter:06d}'
            self.group_counter += 1
            try:
                observation = self.env.reset(group_id, scene)
            except InvalidScene as error:
                self.resets.append({'episode_id': group_id, 'scene_id': scene.scene_id,
                                    'accepted': False, 'reason': str(error)})
                continue
            self.resets.append({'episode_id': group_id, **self.env.reset_audit,
                                'accepted': True})
            initial = observation_transition(self.env, observation)
            actions = policy.sample_chunk([initial] * self.candidates_per_scene,
                                          self.config.chunk_length,
                                          batch_size=self.config.microbatch_size)
            snapshot = self.env.snapshot()
            try:
                for candidate_index, action_chunk in enumerate(actions):
                    episode_id = f'{group_id}_c{candidate_index:02d}'
                    self.env.restore(snapshot)
                    self.env.episode_id = episode_id
                    self.resets.append({'episode_id': episode_id, 'scene_id': scene.scene_id,
                                        'accepted': True, 'restored_from_snapshot': True,
                                        'rollout_group_id': group_id,
                                        'rollout_candidate_index': candidate_index})
                    step_results = []
                    step_transitions = []
                    initial_image, initial_proprio = initial.image, initial.proprio
                    for action in action_chunk:
                        _, transition = self.env.step(action)
                        step_transitions.append(transition)
                        step_results.append(transition.result)
                        if transition.terminated or transition.truncated:
                            break
                    final = step_results[-1]
                    final_transition = step_transitions[-1]
                    first_transition = step_transitions[0]
                    aggregate = _aggregate_chunk_feedback(step_results)
                    chunk = ChunkTransition(
                        episode_id=episode_id,
                        decision_index=0,
                        image=initial_image,
                        instruction=initial.instruction,
                        proprio=initial_proprio,
                        action_labels=tuple(action_chunk),
                        step_results=tuple(step_results),
                        result=aggregate,
                        terminated=final_transition.terminated,
                        truncated=final_transition.truncated,
                        scene_id=scene.scene_id,
                        safety_failure=final_transition.safety_failure,
                        measurements={
                            'object_before': first_transition.measurements['object_before'],
                            'object_after': final_transition.measurements['object_after'],
                            'goal_xy': first_transition.measurements['goal_xy'],
                            'progress_m': aggregate.progress_m,
                        },
                        rollout_group_id=group_id,
                        rollout_candidate_index=candidate_index,
                    )
                    chunks.append(chunk)
                    self.completed.append({'episode_id': episode_id,
                                           'success': aggregate.success,
                                           'length': len(step_results),
                                           'safety_failure': chunk.safety_failure,
                                           'rollout_group_id': group_id})
            finally:
                self.env.remove_snapshot(snapshot)
        expected = groups * self.candidates_per_scene
        if len(chunks) != expected:
            raise RuntimeError('Chunk collection did not produce requested count')
        return chunks

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
    subtract_noop_baseline = bool(teacher_context and teacher_context.get('subtract_noop_baseline'))
    valid_count=int(valid.sum())
    if feedback_mode=='no_feedback':
        return {'loss':0.,'valid':0,'executed':len(transitions),'gradient_norm':0.,
                'optimizer_steps':0,'seconds':0.,'donor_indices':donors,
                'valid_mask':[False]*len(transitions),'teacher_log_probs':[]}
    if not valid_count:
        raise RuntimeError('No valid transitions; refusing an empty optimizer step')
    start=time.perf_counter()
    teacher=torch.full((len(transitions),5),-float(np.log(5.)),dtype=torch.float32)
    privileged_teacher=torch.full_like(teacher, -float(np.log(5.)))
    noop_baseline=torch.full_like(teacher, -float(np.log(5.)))
    residual=torch.zeros_like(teacher)
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
                baseline_lp=None
            else:
                lp=teacher_fn(selected_transitions,feedback=True,
                              **teacher_kwargs)
                baseline_kwargs=dict(teacher_kwargs)
                if subtract_noop_baseline:
                    baseline_kwargs['feedback_records']=[no_op_directional_feedback(record)
                        for record in teacher_kwargs['feedback_records']]
                baseline_lp=(teacher_fn(selected_transitions,feedback=True,**baseline_kwargs)
                             if subtract_noop_baseline else None)
            if not torch.isfinite(lp).all() or (baseline_lp is not None and not torch.isfinite(baseline_lp).all()):
                raise FloatingPointError('Nonfinite teacher probabilities')
            privileged_teacher[selected]=lp.detach().cpu()
            if baseline_lp is not None:
                noop_baseline[selected]=baseline_lp.detach().cpu()
                student_reference=policy.log_probs(selected_transitions,feedback=False,grad=False)
                corrected, delta=apply_teacher_residual(student_reference,lp,baseline_lp)
                teacher[selected]=corrected.cpu()
                residual[selected]=delta.cpu()
            else:
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
            'teacher_privileged_log_probs':privileged_teacher.tolist() if subtract_noop_baseline else None,
            'teacher_noop_baseline_log_probs':noop_baseline.tolist() if subtract_noop_baseline else None,
            'teacher_residual_log_probs':residual.tolist() if subtract_noop_baseline else None,
            'subtract_noop_baseline':subtract_noop_baseline,
            'mean_abs_teacher_residual':float(residual[valid].abs().mean()) if subtract_noop_baseline else None,
            'mean_abs_teacher_residual_by_action':residual[valid].abs().mean(0).tolist() if subtract_noop_baseline else None,
            'mean_teacher_student_total_variation':float(tv[valid].mean()),
            'mean_teacher_entropy':float(entropy[valid].mean()),
            'teacher_argmax_counts':{str(i):int((teacher_argmax[valid]==i).sum()) for i in range(5)},
            'teacher_oracle_accuracy':float((teacher_argmax[oracle_valid]==oracle[oracle_valid]).mean()) if oracle_valid.any() else None,
            'teacher_repeat_positive_accuracy':float((teacher_argmax[positive]==attempted[positive]).mean()) if positive.any() else None,
            'teacher_reject_negative_accuracy':float((teacher_argmax[negative]!=attempted[negative]).mean()) if negative.any() else None}


def sdpo_chunk_update(policy, chunks, microbatch_size=4, teacher_context=None):
    """Distill feedback-conditioned teacher distributions at every chunk position."""
    if not chunks or microbatch_size < 1:
        raise ValueError('Need chunks and positive microbatch size')
    if teacher_context is None:
        raise ValueError('Chunk SDPO requires an explicit chunk teacher context')
    policy.model.eval()
    records = teacher_context['records']
    records_by_position = teacher_context.get('records_by_position')
    valid = np.asarray(teacher_context['valid_mask'], dtype=bool)
    action_chunks = [tuple(chunk.action_labels) for chunk in chunks]
    solution_actions = teacher_context.get('solution_actions')
    include_outcome_feedback = teacher_context.get('include_outcome_feedback', True)
    chunk_length = len(action_chunks[0])
    if records_by_position is None:
        records_by_position = [records for _ in range(chunk_length)]
    if len(records_by_position) != len(chunks):
        raise ValueError('Position feedback must align with chunks')
    position_records = [
        [records_by_position[index][position] for index in range(len(chunks))]
        for position in range(chunk_length)
    ]
    action_mask = torch.as_tensor(
        [chunk.valid_action_mask for chunk in chunks], dtype=torch.bool)
    sample_mask = torch.as_tensor(valid, dtype=torch.bool).unsqueeze(1)
    mask = action_mask & sample_mask
    if not bool(mask.any()):
        raise RuntimeError('No valid chunk actions; refusing an empty optimizer step')
    subtract_noop_baseline = bool(teacher_context.get('subtract_noop_baseline'))
    start = time.perf_counter()
    with torch.no_grad():
        teacher_fn = getattr(policy, 'teacher_chunk_log_probs', None)
        if teacher_fn is None:
            raise RuntimeError('Chunk policy must provide teacher_chunk_log_probs')
        privileged = teacher_fn(chunks, action_chunks, feedback=True,
                                feedback_records=records,
                                feedback_records_by_position=position_records,
                                solution_actions=solution_actions,
                                include_outcome_feedback=include_outcome_feedback,
                                batch_size=microbatch_size)
        baseline = None
        if subtract_noop_baseline:
            baseline_records = teacher_context.get('baseline_records')
            baseline_records_by_position = teacher_context.get('baseline_records_by_position')
            if baseline_records is None:
                raise ValueError('Residual chunk context missing baseline records')
            baseline = teacher_fn(chunks, action_chunks, feedback=True,
                                  feedback_records=baseline_records,
                                  feedback_records_by_position=(
                                      [[baseline_records_by_position[index][position]
                                        for index in range(len(chunks))]
                                       for position in range(chunk_length)]
                                      if baseline_records_by_position is not None else None),
                                  solution_actions=solution_actions,
                                  include_outcome_feedback=include_outcome_feedback,
                                  batch_size=microbatch_size)
            student_reference = policy.chunk_log_probs(
                chunks, action_chunks, feedback=False, grad=False,
                batch_size=microbatch_size)
            target = torch.log_softmax(
                student_reference + privileged - baseline, dim=-1).detach().cpu()
            residual = (privileged - baseline).detach().cpu()
        else:
            target = privileged.detach().cpu()
            residual = None
    policy.optimizer.zero_grad(set_to_none=True)
    total_loss = 0.
    try:
        for start_index in range(0, len(chunks), microbatch_size):
            end_index = start_index + microbatch_size
            student = policy.chunk_log_probs(
                chunks[start_index:end_index],
                action_chunks[start_index:end_index], feedback=False, grad=True)
            q = target[start_index:end_index].to(policy.device)
            local_mask = mask[start_index:end_index].to(policy.device)
            loss_name = getattr(getattr(policy, 'config', None), 'distillation_loss', 'js')
            loss_function = symmetric_js_loss if loss_name == 'js' else reverse_kl_loss
            loss = loss_function(student, q, local_mask) * int(local_mask.sum()) / int(mask.sum())
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite chunk student loss')
            loss.backward()
            total_loss += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(list(policy.trainable_parameters()), 1.,
                                              error_if_nonfinite=True)
        policy.optimizer.step()
        if hasattr(policy, 'update_teacher_ema'):
            policy.update_teacher_ema()
    except Exception:
        policy.optimizer.zero_grad(set_to_none=True)
        raise
    policy.optimizer.zero_grad(set_to_none=True)
    target_probs = target.exp()
    with torch.no_grad():
        student_probs = policy.chunk_log_probs(
            chunks, action_chunks, feedback=False, grad=False).exp().cpu()
    tv = .5 * (student_probs - target_probs).abs().sum(-1)
    entropy = -(target_probs * target.clamp_min(-100.)).sum(-1)
    selected_tv = tv[mask].mean()
    selected_entropy = entropy[mask].mean()
    return {
        'loss': total_loss,
        'valid': int(valid.sum()),
        'valid_actions': int(mask.sum()),
        'executed_chunks': len(chunks),
        'chunk_length': len(action_chunks[0]),
        'gradient_norm': float(norm),
        'optimizer_steps': 1,
        'seconds': time.perf_counter() - start,
        'valid_action_mask': mask.tolist(),
        'teacher_chunk_log_probs': target.tolist(),
        'teacher_privileged_chunk_log_probs': privileged.detach().cpu().tolist()
        if subtract_noop_baseline else None,
        'teacher_noop_chunk_log_probs': baseline.detach().cpu().tolist()
        if subtract_noop_baseline else None,
        'teacher_residual_chunk_log_probs': residual.tolist()
        if residual is not None else None,
        'subtract_noop_baseline': subtract_noop_baseline,
        'mean_abs_teacher_residual': float(residual[mask].abs().mean())
        if residual is not None else None,
        'mean_teacher_student_total_variation': float(selected_tv),
        'mean_teacher_entropy': float(selected_entropy),
    }
