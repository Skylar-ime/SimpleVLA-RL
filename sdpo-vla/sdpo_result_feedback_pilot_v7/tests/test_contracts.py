from dataclasses import replace
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from sdpo_result_feedback.feedback import (FeedbackRecord, build_feedback_text,
                                           causal_feedback, directional_feedback,
                                           select_feedback,
                                           wrong_directional_feedback)
from sdpo_result_feedback.env_adapter import PandaPushOutcomePilot
from sdpo_result_feedback.trainer import GroupedRolloutCollector,TrainingConfig,build_group_teacher_context,sdpo_update
from sdpo_result_feedback.policy import QwenActionPolicy
from sdpo_result_feedback.scenes import create_directional_calibrated_scene_splits,create_directional_local_scene_splits,create_directional_long_scene_splits,create_directional_scene_splits,create_mechanism_scene_splits,create_scene_splits
from sdpo_result_feedback.losses import symmetric_js_loss


def test_replay_and_preaction_image():
    e=PandaPushOutcomePilot(max_decisions=1)
    try:
        before=e.reset('replay_contract')
        snapshot=e.snapshot()
        after,t=e.step('A')
        assert np.array_equal(t.image,before['image'])
        assert np.array_equal(t.proprio,before['proprio'])
        assert t.truncated or t.terminated
        json.dumps(t.as_json_record())
        with pytest.raises(RuntimeError): e.step('A')
        restored=e.restore(snapshot)
        assert np.array_equal(restored['image'],before['image'])
        repeated,t2=e.step('A')
        assert np.array_equal(repeated['image'],after['image'])
        assert abs(t.distance_after_m-t2.distance_after_m)<1e-10
        e.remove_snapshot(snapshot)
    finally:
        e.close()


def test_grouped_collector_reuses_one_exact_initial_state(tmp_path):
    scenes=create_directional_local_scene_splits(tmp_path/'grouped_scenes.json',
                                                 {'train':1,'dev':0,'test':0})['train']
    config=TrainingConfig(transitions_per_update=4,microbatch_size=2,
                          parallel_envs=1,max_decisions=1,seed=11)

    class FixedPolicy:
        def sample_batch(self, inputs):
            return ['A','C','D','B'][:len(inputs)]

    collector=GroupedRolloutCollector(config,scenes,candidates_per_scene=4)
    try:
        transitions=collector.collect(FixedPolicy())
        assert len({transition.episode_id for transition in transitions})==4
        assert all(np.array_equal(transition.image,transitions[0].image)
                   for transition in transitions)
        assert all(np.array_equal(transition.proprio,transitions[0].proprio)
                   for transition in transitions)
        assert all(transition.rollout_group_id==transitions[0].rollout_group_id
                   for transition in transitions)
        assert sum(row.get('restored_from_snapshot',False) for row in collector.resets)==4
    finally:
        collector.close()


def fake_transition(episode,action,result,group=None):
    return SimpleNamespace(episode_id=episode,action_label=action,result=result,safety_failure=None,
                           rollout_group_id=group)


def test_shuffled_no_cross_action_or_episode_and_no_outcome_mutation():
    records=[FeedbackRecord(False,'none','unchanged',False),FeedbackRecord(True,'detected','closer',False)]
    ts=[fake_transition('a','A',records[0]),fake_transition('b','A',records[1]),
        fake_transition('c','B',records[0]),fake_transition('d','A',FeedbackRecord(True,'detected','closer',True))]
    selected,valid,donors=select_feedback(ts,'shuffled')
    assert donors==[1,0,None,None]
    assert valid.all()
    assert selected[0]==records[1] and selected[1]==records[0]
    assert selected[2].success is False and selected[2].contact is None
    assert selected[3].success is True
    assert ts[0].result==records[0]


def test_causal_shuffled_preserves_action_but_changes_outcome_attribution():
    records=[FeedbackRecord(False,'none','unchanged',False),FeedbackRecord(True,'detected','closer',False)]
    ts=[fake_transition('a','A',records[0]),fake_transition('b','D',records[1])]
    selected,valid,donors=select_feedback(ts,'causal_shuffled',seed=3)
    assert valid.all() and donors==[1,0]
    assert selected[0].goal_progress=='closer'
    assert 'PREVIOUS_RESPONSE=A' in selected[0].interpretation
    assert 'OUTCOME_VERDICT=ACCEPT' in selected[0].interpretation
    assert selected[1].goal_progress=='unchanged'
    assert 'PREVIOUS_RESPONSE=D' in selected[1].interpretation
    assert 'OUTCOME_VERDICT=REJECT' in selected[1].interpretation


def test_causal_shuffled_refuses_same_verdict_donor():
    records=[FeedbackRecord(True,'detected','closer',False,.01,.01,.1,.09),
             FeedbackRecord(True,'detected','closer',False,.02,.02,.1,.08),
             FeedbackRecord(False,'none','unchanged',False,0.,0.,.1,.1)]
    ts=[fake_transition(str(i),'A',record) for i,record in enumerate(records)]
    selected,valid,donors=select_feedback(ts,'causal_shuffled',seed=4)
    assert valid.all()
    assert donors[0]==2 and donors[1]==2
    assert donors[2] in {0,1}
    assert all(('OUTCOME_VERDICT=ACCEPT' in selected[i].interpretation) !=
               ('OUTCOME_VERDICT=ACCEPT' in causal_feedback(ts[i].action_label,ts[i].result).interpretation)
               for i in range(3))


def test_grouped_sdpo_uses_only_same_state_self_success_as_demonstration():
    transitions=[
        fake_transition('a','C',FeedbackRecord(True,'detected','closer',False),'g0'),
        fake_transition('b','A',FeedbackRecord(False,'none','unchanged',False),'g0'),
        fake_transition('c','B',FeedbackRecord(False,'none','unchanged',False),'g1'),
        fake_transition('d','D',FeedbackRecord(False,'none','unchanged',False),'g1'),
    ]
    context=build_group_teacher_context(transitions,'solution_only',seed=7)
    assert context['groups_with_witness']==1
    assert context['solution_actions']==[None,'C',None,None]
    assert context['valid_mask'].tolist()==[False,True,False,False]
    assert not context['include_outcome_feedback']
    assert all(record.contact is None for record in context['records'])
    rich=build_group_teacher_context(transitions,'solution_plus_causal',seed=7)
    assert rich['solution_actions']==[None,'C',None,None]
    assert rich['valid_mask'].tolist()==[False,True,True,True]
    assert rich['records'][1].interpretation.startswith('PREVIOUS_RESPONSE=A')


def test_hard_shuffled_feedback_is_different_and_cross_episode():
    records=[FeedbackRecord(False,'none','unchanged',False,0.,0.,.1,.1),
             FeedbackRecord(True,'detected','closer',False,.01,.01,.1,.09),
             FeedbackRecord(True,'detected','farther',False,-.01,.01,.1,.11)]
    ts=[fake_transition(str(i),'A',record) for i,record in enumerate(records)]
    selected,valid,donors=select_feedback(ts,'hard_shuffled',seed=4)
    assert valid.all()
    assert all(donor is not None for donor in donors)
    assert all(selected[i]!=records[i] for i in range(len(records)))
    assert all(ts[donor].episode_id!=ts[i].episode_id for i,donor in enumerate(donors))


def test_directional_feedback_reports_measured_vector_without_action_label():
    transition = fake_transition('0', 'A', FeedbackRecord(False, 'none', 'unchanged', False))
    transition.measurements = {
        'object_before': [0.01, -0.02, 0.02],
        'goal_xy': [-0.14, -0.02],
    }
    feedback = directional_feedback('A', transition)
    assert feedback.target_direction == '-x'
    assert feedback.target_dx_m == pytest.approx(-0.15)
    assert feedback.target_dy_m == pytest.approx(0.0)
    assert 'Dominant remaining direction: -x' in build_feedback_text('A', feedback)


def test_wrong_directional_feedback_flips_measurement_only():
    transition = SimpleNamespace(
        result=FeedbackRecord(True, 'detected', 'closer', False),
        measurements={'object_before': [0.0, 0.0, 0.02], 'goal_xy': [0.1, 0.02]},
    )
    feedback = wrong_directional_feedback('A', transition)
    assert feedback.target_dx_m == -0.1
    assert feedback.target_dy_m == -0.02
    assert feedback.target_direction == '-x'


def test_wrong_directional_condition_is_registered():
    transition = fake_transition('0', 'A', FeedbackRecord(False, 'none', 'unchanged', False), 'g0')
    transition.measurements = {
        'object_before': [0.0, 0.0, 0.02],
        'goal_xy': [0.1, 0.02],
    }
    from sdpo_result_feedback.sdpo_context import build_sdpo_context
    context = build_sdpo_context([transition], 'sdpo_wrong_directional')
    assert context['records'][0].target_direction == '-x'


def test_teacher_stop_gradient():
    p=torch.randn(7,5,requires_grad=True)
    q=torch.randn(7,5,requires_grad=True)
    loss=symmetric_js_loss(p.log_softmax(-1),q.log_softmax(-1))
    loss.backward()
    assert p.grad is not None and p.grad.abs().sum()>0
    assert q.grad is None


class TinyPolicy:
    def __init__(self):
        self.device=torch.device('cpu')
        self.model=torch.nn.Linear(2,5,bias=False)
        with torch.no_grad(): self.model.weight.copy_(torch.arange(10).reshape(5,2)*.03)
        self.optimizer=torch.optim.SGD(self.model.parameters(),lr=.01)
        self.teacher_weights=[]
        self.steps=0
        original=self.optimizer.step
        def step(*a,**kw):
            self.steps+=1
            return original(*a,**kw)
        self.optimizer.step=step
    def trainable_parameters(self): return self.model.parameters()
    def log_probs(self,transitions,feedback=False,grad=False,feedback_records=None):
        x=torch.tensor([[float(t.episode_id),float(feedback)] for t in transitions])
        if feedback: self.teacher_weights.append(self.model.weight.detach().clone())
        return self.model(x).log_softmax(-1)


def test_microbatch_gradient_equals_full_batch_and_one_step():
    ts=[fake_transition(str(i),'A',FeedbackRecord(True,'detected','closer',False)) for i in range(7)]
    p,q=TinyPolicy(),TinyPolicy()
    a=sdpo_update(p,ts,microbatch_size=2)
    b=sdpo_update(q,ts,microbatch_size=7)
    assert p.steps==q.steps==1
    assert abs(a['loss']-b['loss'])<1e-7
    assert torch.allclose(p.model.weight,q.model.weight,atol=1e-7)
    assert all(torch.equal(w,p.teacher_weights[0]) for w in p.teacher_weights)


def test_student_prompt_cannot_leak_results_or_action():
    t=SimpleNamespace(proprio=np.zeros(3),instruction='Push the green cube.',
                      action_label='A',result=FeedbackRecord(True,'detected','closer',False))
    student=QwenActionPolicy._prompt(t)
    t.action_label='D';t.result=FeedbackRecord(False,'none','farther',True)
    assert QwenActionPolicy._prompt(t)==student
    assert 'Retrospective' not in student


def test_teacher_self_demonstration_stays_out_of_student_prompt():
    t=SimpleNamespace(proprio=np.zeros(3),instruction='Push the green cube.',
                      action_label='A',result=FeedbackRecord(True,'detected','closer',False))
    student=QwenActionPolicy._prompt(t)
    teacher=QwenActionPolicy._prompt(t,feedback=True,solution_action='C',include_outcome_feedback=False)
    assert 'self-generated solution' not in student
    assert 'Correct self-generated solution' in teacher
    assert '\nC\n' in teacher


def test_scene_split_disjoint_and_reproducible(tmp_path):
    p=tmp_path/'scenes.json'
    a=create_scene_splits(p,{'train':10,'dev':4,'test':4})
    b=create_scene_splits(p,{'train':10,'dev':4,'test':4})
    assert a==b
    keys=[tuple(s.object_xy+s.goal_xy+s.ee_xy) for group in a.values() for s in group]
    assert len(keys)==len(set(keys))
    assert {s.split for s in a['train']}=={'train'}


def test_mechanism_scene_split_is_near_contact_and_target_independent(tmp_path):
    scenes=create_mechanism_scene_splits(tmp_path/'mechanism.json',{'train':64,'dev':8,'test':8})
    distances=[]
    relative_signs=[]
    for scene in scenes['train']:
        obj=np.asarray(scene.object_xy)
        goal=np.asarray(scene.goal_xy)-obj
        ee=np.asarray(scene.ee_xy)-obj
        distances.append(np.linalg.norm(ee))
        relative_signs.append(float(np.dot(goal,ee)))
    assert min(distances)>=.044 and max(distances)<=.052
    assert any(v<0 for v in relative_signs) and any(v>0 for v in relative_signs)


def test_directional_scene_split_is_safe_and_action_unlabeled(tmp_path):
    scenes=create_directional_scene_splits(tmp_path/'directional.json',{'train':64,'dev':8,'test':8})
    for scene in scenes['train']:
        obj=np.asarray(scene.object_xy)
        goal=np.asarray(scene.goal_xy)-obj
        ee=np.asarray(scene.ee_xy)-obj
        assert np.isclose(goal[0], 0.) or np.isclose(goal[1], 0.)
        assert np.isclose(ee[0], 0.) or np.isclose(ee[1], 0.)
        assert np.linalg.norm(ee)>=.045 and np.linalg.norm(ee)<=.052
        assert np.dot(goal,ee)<0


def test_directional_long_scene_needs_more_than_one_useful_push(tmp_path):
    scenes=create_directional_long_scene_splits(tmp_path/'directional_long.json',{'train':64,'dev':8,'test':8})
    for scene in scenes['train']:
        obj=np.asarray(scene.object_xy)
        goal=np.asarray(scene.goal_xy)-obj
        ee=np.asarray(scene.ee_xy)-obj
        assert .13<=np.linalg.norm(goal)<=.16
        assert .045<=np.linalg.norm(ee)<=.052
        assert np.dot(goal,ee)<0


def test_directional_calibrated_scene_uses_axis_specific_offsets(tmp_path):
    scenes=create_directional_calibrated_scene_splits(tmp_path/'directional_calibrated.json',
        {'train':256,'dev':16,'test':16})
    directions=set()
    for scene in scenes['train']:
        obj=np.asarray(scene.object_xy)
        goal=np.asarray(scene.goal_xy)-obj
        ee=np.asarray(scene.ee_xy)-obj
        axis=0 if abs(goal[0])>abs(goal[1]) else 1
        directions.add((axis,int(np.sign(goal[axis]))))
        expected=.040 if axis==0 else .055
        assert np.isclose(np.linalg.norm(ee),expected)
        assert .13<=np.linalg.norm(goal)<=.16
        assert np.dot(goal,ee)<0
    assert directions=={(0,-1),(0,1),(1,-1),(1,1)}


def test_normalized_action_scales_to_two_centimeters():
    from sdpo_result_feedback.actions import action_vector
    e=PandaPushOutcomePilot()
    try:
        e.reset('action_scale_contract')
        before=e.base.robot.get_ee_position().copy()
        captured=[]
        original=e.base.robot.inverse_kinematics
        def capture(link,position,orientation):
            captured.append(np.asarray(position).copy())
            return original(link,position,orientation)
        e.base.robot.inverse_kinematics=capture
        e.base.robot.set_action(action_vector('A'))
        assert np.allclose(captured[-1][:2]-before[:2],[.02,0.],atol=1e-7)
        assert captured[-1][2]==.025
    finally:
        e.close()


def test_js_matches_manual_distribution():
    p=torch.tensor([[.1,.2,.3,.15,.25]],requires_grad=True)
    q=torch.tensor([[.4,.1,.2,.2,.1]],requires_grad=True)
    mixture=(p.detach().numpy()+q.detach().numpy())/2
    manual=.5*np.sum(p.detach().numpy()*np.log(p.detach().numpy()/mixture))+ .5*np.sum(q.detach().numpy()*np.log(q.detach().numpy()/mixture))
    actual=symmetric_js_loss(p.log(),q.log())
    assert abs(float(actual.detach())-manual)<1e-7
    actual.backward()
    assert q.grad is None


def test_teacher_refreshes_next_round_and_rewards_ignored():
    ts=[fake_transition(str(i),'A',FeedbackRecord(True,'detected','closer',False)) for i in range(7)]
    p,q=TinyPolicy(),TinyPolicy()
    a=sdpo_update(p,ts,microbatch_size=2)
    for i,t in enumerate(ts):
        t.reward=(-1)**i*1e8
        t.advantage=1e10
    b=sdpo_update(q,ts,microbatch_size=2)
    assert torch.equal(p.model.weight,q.model.weight)
    previous=p.model.weight.detach().clone()
    count=len(p.teacher_weights)
    sdpo_update(p,ts,microbatch_size=2)
    assert torch.equal(p.teacher_weights[count],previous)
    assert not torch.equal(p.teacher_weights[0],previous)


def test_identical_prompt_teacher_has_zero_gradient():
    class IdenticalPrompt(TinyPolicy):
        def log_probs(self,transitions,feedback=False,grad=False,feedback_records=None):
            return super().log_probs(transitions,feedback=False,grad=grad)
    p=IdenticalPrompt()
    original=p.model.weight.detach().clone()
    ts=[fake_transition(str(i),'A',FeedbackRecord(True,'detected','closer',False)) for i in range(7)]
    result=sdpo_update(p,ts,microbatch_size=2)
    assert abs(result['loss'])<1e-7
    assert result['gradient_norm']<1e-7
    assert torch.allclose(p.model.weight,original,atol=1e-8)
