"""Calibrated PandaPush wrapper. Hidden geometry is never returned as actor input."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any
import gymnasium as gym
import numpy as np
import panda_gym  # noqa: F401; registers Gym environments.
from PIL import Image, ImageDraw
from .actions import DECISION_SIM_STEPS, RESULT_DEADBAND_M, action_vector
from .feedback import FeedbackRecord
from .scenes import Scene, propose_scene

EE_HEIGHT_M = .025
CAMERA = dict(target_position=[0., 0., .02], distance=.8, yaw=180., pitch=-30., roll=0., width=448, height=448)


@dataclass
class Transition:
    episode_id: str
    decision_index: int
    image: np.ndarray
    instruction: str
    proprio: np.ndarray
    action_label: str
    result: FeedbackRecord | None
    actual_ee_delta_m: list[float]
    distance_before_m: float
    distance_after_m: float
    terminated: bool
    truncated: bool
    scene_id: str = ''
    safety_failure: str | None = None
    measurements: dict = field(default_factory=dict)
    rollout_group_id: str | None = None
    rollout_candidate_index: int | None = None

    def as_json_record(self) -> dict[str, Any]:
        # Do not copy every 448x448 image while serializing scalar records.
        record = {k: v for k, v in vars(self).items() if k not in {'image', 'proprio', 'result'}}
        return {**record, 'proprio': self.proprio.tolist(),
                'result': None if self.result is None else self.result.as_dict()}


class InvalidScene(ValueError):
    pass


class PandaPushOutcomePilot:
    def __init__(self, max_decisions=32, seed=11):
        self.env = gym.make('PandaPush-v3', render_mode='rgb_array', renderer='Tiny',
            reward_type='sparse', control_type='ee', render_width=448, render_height=448,
            render_target_position=np.array(CAMERA['target_position']), render_distance=CAMERA['distance'],
            render_yaw=CAMERA['yaw'], render_pitch=CAMERA['pitch'], render_roll=CAMERA['roll'])
        self.base = self.env.unwrapped
        self.physics = self.base.sim.physics_client
        self.robot_id = self.base.sim._bodies_idx['panda']
        self.object_id = self.base.sim._bodies_idx['object']
        self.base.task.distance_threshold = .02
        self.max_decisions, self.seed = max_decisions, seed
        self.episode_id, self.decision_index = '', 0
        self.last_obs, self.scene = None, None
        self.done = False
        self.control_target = None
        original_control = self.base.robot.control_joints
        def tracked_control(target_angles):
            self.control_target = np.asarray(target_angles).copy()
            original_control(target_angles)
        self.base.robot.control_joints = tracked_control
        def fixed_height_target(displacement):
            # Identical for all policies: no target/object-dependent action correction.
            target = self.base.robot.get_ee_position() + np.asarray(displacement) * .05
            target[2] = EE_HEIGHT_M
            return self.base.robot.inverse_kinematics(11,target,np.array([1.,0.,0.,0.]))[:7]
        self.base.robot.ee_displacement_to_target_arm_angles = fixed_height_target
        self.physics.changeVisualShape(self.base.sim._bodies_idx['target'], -1, rgbaColor=[.15,.3,1.,.65])
        self.view = np.array(self.physics.computeViewMatrixFromYawPitchRoll(
            CAMERA['target_position'], CAMERA['distance'], CAMERA['yaw'], CAMERA['pitch'], CAMERA['roll'], 2)).reshape(4,4,order='F')
        self.projection = np.array(self.physics.computeProjectionMatrixFOV(60.,1.,.1,100.)).reshape(4,4,order='F')

    def reset(self, episode_id=None, scene: Scene | None = None):
        scene = scene or propose_scene(episode_id or f'scene_{self.seed}', seed=self.seed)
        self.env.reset(seed=scene.seed)
        self.episode_id, self.scene = episode_id or scene.scene_id, scene
        self.decision_index, self.done = 0, False
        self.base.task.goal = np.array([*scene.goal_xy, .02])
        self.base.sim.set_base_pose('target', self.base.task.goal, np.array([0.,0.,0.,1.]))
        self.base.sim.set_base_pose('object', np.array([*scene.object_xy,.02]), np.array([0.,0.,0.,1.]))
        self.physics.resetBaseVelocity(self.object_id, [0.,0.,0.], [0.,0.,0.])
        position = np.array([*scene.ee_xy, EE_HEIGHT_M])
        for _ in range(25):
            arm = self.base.robot.inverse_kinematics(11, position, np.array([1.,0.,0.,0.]))[:7]
            joints = np.r_[arm,0.,0.]
            self.base.robot.set_joint_angles(joints)
        self.base.robot.control_joints(joints)
        self.physics.performCollisionDetection()
        overlaps = [c for c in self.physics.getClosestPoints(self.robot_id,self.object_id,0.) if c[8] < -1e-6]
        if overlaps:
            self.last_obs = None
            raise InvalidScene(f'{scene.scene_id}: robot-object overlap')
        before = self.base.task.get_achieved_goal().copy()
        for _ in range(10):
            self.base.sim.step()
        after = self.base.task.get_achieved_goal().copy()
        drift = float(np.linalg.norm(after[:2]-before[:2]))
        ee_error = float(np.linalg.norm(self.base.robot.get_ee_position()-position))
        if drift > .002 or ee_error > .003 or self._distance() < .02:
            self.last_obs = None
            raise InvalidScene(f'{scene.scene_id}: invalid reset drift={drift} ee_error={ee_error}')
        self.reset_audit = {'scene':scene.as_dict(), 'settle_drift_m':drift, 'ee_position_error_m':ee_error,
                            'joint_angles':joints.tolist(), 'settle_sim_steps':10}
        self.last_obs = self.base._get_obs()
        return self._observation()

    def _camera_matrices(self):
        view = np.array(self.physics.computeViewMatrixFromYawPitchRoll(
            CAMERA['target_position'], CAMERA['distance'], CAMERA['yaw'], CAMERA['pitch'], CAMERA['roll'], 2)).reshape(4,4,order='F')
        return view, self.projection

    def project(self, xyz):
        view, projection = self._camera_matrices()
        clip = projection @ view @ np.r_[xyz,1.]
        ndc = clip[:3] / clip[3]
        return np.array([(ndc[0]+1)*224, (1-ndc[1])*224])

    def _image(self):
        image = Image.fromarray(np.asarray(self.base.sim.render(width=448,height=448,target_position=np.array(CAMERA['target_position']),distance=CAMERA['distance'],yaw=CAMERA['yaw'],pitch=CAMERA['pitch'],roll=CAMERA['roll'])))
        draw = ImageDraw.Draw(image)
        # Fixed axis legend: directions computed using the rendering matrices.
        origin = self.project([0.,0.,.02])
        center = np.array([55.,385.])
        for endpoint, label, color in [([.06,0.,.02],'+x A',(170,20,20)),([0.,.06,.02],'+y C',(15,65,175))]:
            d = self.project(endpoint)-origin
            d = d / np.linalg.norm(d) * 32
            end = center + d
            draw.line([tuple(center),tuple(end)], fill=color, width=3)
            u=d/np.linalg.norm(d); v=np.array([-u[1],u[0]])
            draw.polygon([tuple(end),tuple(end-7*u+4*v),tuple(end-7*u-4*v)], fill=color)
            draw.text(tuple(end + np.array([3.,-6.])), label, fill=color)
        return np.asarray(image).copy()

    def _observation(self):
        if self.last_obs is None:
            raise RuntimeError('reset must succeed before observation')
        return {'image':self._image(), 'instruction':'Push the green cube to the blue target.',
                'proprio':np.asarray(self.base.robot.get_ee_position(),dtype=np.float32).copy()}

    def _distance(self):
        return float(np.linalg.norm(self.base.task.get_achieved_goal()-self.base.task.get_goal()))

    def _contact(self):
        return any(c[3] in {9,10,11} for c in self.physics.getContactPoints(self.robot_id,self.object_id))

    def step(self, action_label):
        if self.last_obs is None or self.done:
            raise RuntimeError('reset required before step (including after termination)')
        before = self._observation()
        distance = self._distance()
        obj = self.base.task.get_achieved_goal().copy()
        goal = self.base.task.get_goal().copy()
        ee = self.base.robot.get_ee_position().copy()
        self.base.robot.set_action(action_vector(action_label))
        contact = False
        for _ in range(DECISION_SIM_STEPS * self.base.sim.n_substeps):
            self.physics.stepSimulation()
            contact = contact or self._contact()
        self.last_obs = self.base._get_obs()
        obj_after = self.base.task.get_achieved_goal().copy()
        ee_after = self.base.robot.get_ee_position().copy()
        distance_after = self._distance()
        movement = float(np.linalg.norm(obj_after[:2]-obj[:2]))
        delta = distance-distance_after
        success = bool(distance_after < .02)
        failure = None
        if not np.isfinite(np.r_[obj_after,ee_after,distance_after]).all():
            failure = 'nonfinite_state'
        elif abs(obj_after[0])>.27 or abs(obj_after[1])>.30 or obj_after[2]<0.:
            failure = 'object_out_of_workspace'
        elif ee_after[0]<-.32 or ee_after[0]>.25 or abs(ee_after[1])>.30:
            failure = 'end_effector_out_of_workspace'
        if failure:
            success = False
        self.decision_index += 1
        terminated = success or failure is not None
        truncated = self.decision_index >= self.max_decisions and not terminated
        self.done = terminated or truncated
        result = FeedbackRecord(contact,'detected' if movement>RESULT_DEADBAND_M else 'none',
            'closer' if delta>RESULT_DEADBAND_M else 'farther' if delta < -RESULT_DEADBAND_M else 'unchanged',success,
            progress_m=delta, object_motion_m=movement, distance_before_m=distance,
            distance_after_m=distance_after)
        transition = Transition(self.episode_id,self.decision_index-1,before['image'],before['instruction'],
            before['proprio'],action_label,result,(ee_after-ee).tolist(),distance,distance_after,terminated,truncated,
            self.scene.scene_id,failure,{'object_before':obj.tolist(),'object_after':obj_after.tolist(),
                'goal_xy':goal[:2].tolist(),
                'object_motion_xy_m':movement,'progress_m':delta,'simulated_time_s':self.decision_index*.16,
                'requested_action':action_vector(action_label).tolist(),'applied_action':action_vector(action_label).tolist()})
        return self._observation(), transition

    def snapshot(self):
        if self.last_obs is None:
            raise RuntimeError('reset required')
        return {'physics_id': self.base.sim.save_state(), 'goal':self.base.task.goal.copy(),
                'base_rng':deepcopy(self.base.np_random.bit_generator.state),
                'task_rng':deepcopy(self.base.task.np_random.bit_generator.state),
                'control_target':self.control_target.copy(), 'elapsed_steps':self.env._elapsed_steps,
                'python':deepcopy({k:getattr(self,k) for k in ['episode_id','scene','decision_index','done','last_obs','reset_audit']})}

    def restore(self, snapshot):
        self.base.sim.restore_state(snapshot['physics_id'])
        self.base.task.goal = snapshot['goal'].copy()
        self.base.np_random.bit_generator.state = deepcopy(snapshot['base_rng'])
        self.base.task.np_random.bit_generator.state = deepcopy(snapshot['task_rng'])
        self.base.robot.control_joints(snapshot['control_target'])
        self.env._elapsed_steps = snapshot['elapsed_steps']
        for key,value in snapshot['python'].items():
            setattr(self,key,deepcopy(value))
        return self._observation()

    def remove_snapshot(self,snapshot):
        self.base.sim.remove_state(snapshot['physics_id'])

    def close(self):
        self.env.close()
