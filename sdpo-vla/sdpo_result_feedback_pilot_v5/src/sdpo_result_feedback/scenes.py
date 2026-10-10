"""Deterministic, parameter-disjoint scene proposals; no target-conditioned EE side."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Scene:
    scene_id: str
    split: str
    seed: int
    object_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    ee_xy: tuple[float, float]

    def as_dict(self):
        return asdict(self)


def seed_from_name(name: str, seed: int = 11) -> int:
    return int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:4], 'little')


def propose_scene(scene_id: str, split: str = 'development', seed: int = 11) -> Scene:
    scene_seed = seed_from_name(scene_id, seed)
    rng = np.random.default_rng(scene_seed)
    for _ in range(10000):
        obj = rng.uniform(-.08, .08, 2)
        angle = rng.uniform(-np.pi, np.pi)
        goal = obj + rng.uniform(.10, .14) * np.array([np.cos(angle), np.sin(angle)])
        angle = rng.uniform(-np.pi, np.pi)  # Independent of goal direction.
        ee = obj + rng.uniform(.055, .085) * np.array([np.cos(angle), np.sin(angle)])
        if np.max(np.abs(goal)) <= .23:
            return Scene(scene_id, split, scene_seed, tuple(obj), tuple(goal), tuple(ee))
    raise RuntimeError('Scene sampler exhausted')


def propose_mechanism_scene(scene_id: str, split: str = 'development', seed: int = 11) -> Scene:
    """Short-horizon diagnostic scene with target-independent near-contact EE placement."""
    scene_seed = seed_from_name(scene_id, seed)
    rng = np.random.default_rng(scene_seed)
    for _ in range(10000):
        obj = rng.uniform(-.08, .08, 2)
        goal_angle = rng.uniform(-np.pi, np.pi)
        goal = obj + rng.uniform(.07, .09) * np.array([np.cos(goal_angle), np.sin(goal_angle)])
        ee_angle = rng.uniform(-np.pi, np.pi)
        ee = obj + rng.uniform(.044, .052) * np.array([np.cos(ee_angle), np.sin(ee_angle)])
        if np.max(np.abs(goal)) <= .20:
            return Scene(scene_id, split, scene_seed, tuple(obj), tuple(goal), tuple(ee))
    raise RuntimeError('Mechanism scene sampler exhausted')


def propose_directional_scene(scene_id: str, split: str = 'development', seed: int = 11) -> Scene:
    scene_seed = seed_from_name(scene_id, seed)
    rng = np.random.default_rng(scene_seed)
    directions = np.asarray([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    for _ in range(10000):
        obj = rng.uniform(-.06, .06, 2)
        direction = directions[int(rng.integers(0, len(directions)))]
        goal = obj + direction * rng.uniform(.07, .09)
        ee = obj - direction * rng.uniform(.045, .052)
        if np.max(np.abs(goal)) <= .20 and np.max(np.abs(ee)) <= .20:
            return Scene(scene_id, split, scene_seed, tuple(obj), tuple(goal), tuple(ee))
    raise RuntimeError('Directional scene sampler exhausted')


def propose_directional_long_scene(scene_id: str, split: str = 'development', seed: int = 11) -> Scene:
    """Axis-aligned scene where one useful push gives progress without instant success."""
    scene_seed = seed_from_name(scene_id, seed)
    rng = np.random.default_rng(scene_seed)
    directions = np.asarray([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    for _ in range(10000):
        obj = rng.uniform(-.04, .04, 2)
        direction = directions[int(rng.integers(0, len(directions)))]
        goal = obj + direction * rng.uniform(.13, .16)
        ee = obj - direction * rng.uniform(.045, .052)
        if np.max(np.abs(goal)) <= .22 and np.max(np.abs(ee)) <= .20:
            return Scene(scene_id, split, scene_seed, tuple(obj), tuple(goal), tuple(ee))
    raise RuntimeError('Long directional scene sampler exhausted')


def propose_directional_calibrated_scene(scene_id: str, split: str = 'development', seed: int = 11) -> Scene:
    """Axis-aligned scene with direction-specific, empirically calibrated contact geometry.

    Panda arm kinematics are not symmetric across world axes. A shared cube-to-EE
    offset made y pushes immediately informative while x pushes often only moved
    the gripper toward the cube. These offsets produce measurable positive progress
    for one correct action in every direction without completing the task in one step.
    """
    scene_seed = seed_from_name(scene_id, seed)
    rng = np.random.default_rng(scene_seed)
    directions = np.asarray([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    for _ in range(10000):
        obj = rng.uniform(-.04, .04, 2)
        direction = directions[int(rng.integers(0, len(directions)))]
        goal = obj + direction * rng.uniform(.13, .16)
        contact_offset = .040 if direction[0] else .055
        ee = obj - direction * contact_offset
        if np.max(np.abs(goal)) <= .22 and np.max(np.abs(ee)) <= .20:
            return Scene(scene_id, split, scene_seed, tuple(obj), tuple(goal), tuple(ee))
    raise RuntimeError('Calibrated directional scene sampler exhausted')


def create_mechanism_scene_splits(path: Path, counts=None, seed=4171):
    counts = counts or {'train': 2048, 'dev': 128, 'test': 256}
    scenes = {split: [propose_mechanism_scene(f'{split}_{i:06d}', split, seed) for i in range(n)]
              for split, n in counts.items()}
    keys = [tuple(round(v, 12) for v in (s.object_xy + s.goal_xy + s.ee_xy)) for group in scenes.values() for s in group]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate mechanism scene parameters across splits')
    payload = {split: [s.as_dict() for s in group] for split, group in scenes.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(payload)):
            raise FileExistsError('Refusing to replace a different frozen mechanism scene split')
    else:
        path.write_text(json.dumps(payload, indent=2) + '\n')
    return scenes


def create_directional_scene_splits(path: Path, counts=None, seed=4171):
    counts = counts or {'train': 2048, 'dev': 128, 'test': 256}
    scenes = {split: [propose_directional_scene(f'{split}_{i:06d}', split, seed) for i in range(n)]
              for split, n in counts.items()}
    keys = [tuple(round(v, 12) for v in (s.object_xy + s.goal_xy + s.ee_xy)) for group in scenes.values() for s in group]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate directional scene parameters across splits')
    payload = {split: [s.as_dict() for s in group] for split, group in scenes.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(payload)):
            raise FileExistsError('Refusing to replace a different frozen directional scene split')
    else:
        path.write_text(json.dumps(payload, indent=2) + '\n')
    return scenes


def create_directional_long_scene_splits(path: Path, counts=None, seed=4171):
    counts = counts or {'train': 2048, 'dev': 128, 'test': 256}
    scenes = {split: [propose_directional_long_scene(f'{split}_{i:06d}', split, seed) for i in range(n)]
              for split, n in counts.items()}
    keys = [tuple(round(v, 12) for v in (s.object_xy + s.goal_xy + s.ee_xy)) for group in scenes.values() for s in group]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate long directional scene parameters across splits')
    payload = {split: [s.as_dict() for s in group] for split, group in scenes.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(payload)):
            raise FileExistsError('Refusing to replace a different frozen long directional scene split')
    else:
        path.write_text(json.dumps(payload, indent=2) + '\n')
    return scenes


def create_directional_calibrated_scene_splits(path: Path, counts=None, seed=4171):
    counts = counts or {'train': 2048, 'dev': 128, 'test': 256}
    scenes = {split: [propose_directional_calibrated_scene(f'{split}_{i:06d}', split, seed) for i in range(n)]
              for split, n in counts.items()}
    keys = [tuple(round(v, 12) for v in (s.object_xy + s.goal_xy + s.ee_xy)) for group in scenes.values() for s in group]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate calibrated directional scene parameters across splits')
    payload = {split: [s.as_dict() for s in group] for split, group in scenes.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(payload)):
            raise FileExistsError('Refusing to replace a different frozen calibrated directional scene split')
    else:
        path.write_text(json.dumps(payload, indent=2) + '\n')
    return scenes


def create_scene_splits(path: Path, counts=None, seed=4171):
    counts = counts or {'train': 2048, 'dev': 128, 'test': 256}
    scenes = {split: [propose_scene(f'{split}_{i:06d}', split, seed) for i in range(n)]
              for split, n in counts.items()}
    keys = [tuple(round(v, 12) for v in (s.object_xy + s.goal_xy + s.ee_xy)) for group in scenes.values() for s in group]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate scene parameters across splits')
    payload = {split: [s.as_dict() for s in group] for split, group in scenes.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(payload)):
            raise FileExistsError('Refusing to replace a different frozen scene split')
    else:
        path.write_text(json.dumps(payload, indent=2) + '\n')
    return scenes
