"""Measure Molmo2 image pointing on the fixed directional-long dev scenes."""
from collections import Counter
from pathlib import Path
import argparse
import json
import re
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from sdpo_result_feedback.env_adapter import InvalidScene, PandaPushOutcomePilot
from sdpo_result_feedback.scenes import create_directional_long_scene_splits
from sdpo_result_feedback.trainer import observation_transition


POINT_TAG = re.compile(r'<(?:points|point).*?coords="([0-9\t:;, .]+)"', re.IGNORECASE)
POINT_TRIPLE = re.compile(r'(?:^|[\t:;, ])([0-9]+)[ ]+([0-9]{3,4})[ ]+([0-9]{3,4})(?=$|[\t:;, ])')
POINT_PAIR = re.compile(r'(?:^|[\t:;, ])([0-9]{3,4})[ ]+([0-9]{3,4})(?=$|[\t:;, ])')


def mask_bbox(image, kind):
    pixels = image.astype(float)
    if kind == 'cube':
        mask = ((pixels[:, :, 1] > 60)
                & (pixels[:, :, 1] > pixels[:, :, 0] * 1.7)
                & (pixels[:, :, 1] > pixels[:, :, 2] * 1.7))
    else:
        mask = ((pixels[:, :, 2] > 60)
                & (pixels[:, :, 2] > pixels[:, :, 0] * 1.7)
                & (pixels[:, :, 2] > pixels[:, :, 1] * 1.7))
        mask[340:, :130] = False
    y, x = np.where(mask)
    if not len(x):
        return None
    return {
        'bbox': [int(x.min()), int(y.min()), int(x.max()), int(y.max())],
        'centroid': [float(x.mean()), float(y.mean())],
        'pixels': int(len(x)),
    }


def parse_points(text, width, height):
    matches = POINT_TAG.findall(text)
    points = []
    for value in matches:
        triples = POINT_TRIPLE.findall(value)
        if triples:
            points.extend((float(x) / 1000 * width, float(y) / 1000 * height)
                          for _, x, y in triples)
            continue
        points.extend((float(x) / 1000 * width, float(y) / 1000 * height)
                      for x, y in POINT_PAIR.findall(value))
    return points


def point_metrics(point, target):
    if point is None or target is None:
        return {'parsed': False, 'error_pixels': None, 'inside_bbox': False}
    x, y = point
    tx, ty = target['centroid']
    x0, y0, x1, y1 = target['bbox']
    padding = 8
    return {
        'parsed': True,
        'error_pixels': float(np.hypot(x - tx, y - ty)),
        'error_normalized': float(np.hypot(x - tx, y - ty) / np.hypot(448, 448)),
        'inside_bbox': bool(x0 - padding <= x <= x1 + padding and
                            y0 - padding <= y <= y1 + padding),
    }


def geometry_metrics(point, reference, width, height):
    if point is None or reference is None:
        return {'parsed': False, 'error_pixels': None, 'error_normalized': None,
                'within_32px': False}
    error = float(np.hypot(point[0] - reference[0], point[1] - reference[1]))
    return {
        'parsed': True,
        'error_pixels': error,
        'error_normalized': error / np.hypot(width, height),
        'within_32px': error <= 32.,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-id', default='allenai/Molmo2-8B')
    parser.add_argument('--model-revision')
    parser.add_argument('--scene-file', type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        'artifacts/directional_long_scenes_v1.json')
    parser.add_argument('--dev-scenes', type=int, default=64)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'Refusing to overwrite existing output: {args.output}')
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    scenes = create_directional_long_scene_splits(args.scene_file)['dev'][:args.dev_scenes]
    processor_options = {'trust_remote_code': True}
    model_options = {'trust_remote_code': True, 'dtype': 'auto', 'device_map': 'auto'}
    if Path(args.model_id).is_dir():
        processor_options['local_files_only'] = True
        model_options['local_files_only'] = True
    if args.model_revision:
        processor_options['revision'] = args.model_revision
        model_options['revision'] = args.model_revision
    processor = AutoProcessor.from_pretrained(args.model_id, **processor_options)
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_options)
    device = next(model.parameters()).device
    env = PandaPushOutcomePilot(seed=args.seed)
    rows = []
    rejected = []
    start = time.perf_counter()
    try:
        for scene in scenes:
            try:
                observation = env.reset(f'molmo_probe_{scene.scene_id}', scene)
            except InvalidScene as error:
                rejected.append({'scene_id': scene.scene_id, 'reason': str(error)})
                continue
            transition = observation_transition(env, observation)
            image = Image.fromarray(transition.image).convert('RGB')
            targets = {kind: mask_bbox(transition.image, kind)
                       for kind in ('cube', 'target')}
            references = {
                'cube': env.project([*scene.object_xy, .02]).tolist(),
                'target': env.project([*scene.goal_xy, .02]).tolist(),
            }
            scene_row = {'scene_id': scene.scene_id, 'targets': targets,
                         'reference_points': references, 'queries': {}}
            for kind, prompt in {
                'cube': 'Point to the center of the solid green cube. Return one point.',
                'target': 'Point to the center of the translucent blue square target. Return one point.',
            }.items():
                messages = [{'role': 'user', 'content': [
                    {'type': 'text', 'text': prompt}, {'type': 'image', 'image': image}]}]
                inputs = processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_tensors='pt', return_dict=True)
                inputs = {key: value.to(device) if torch.is_tensor(value) else value
                          for key, value in inputs.items()}
                with torch.inference_mode():
                    output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
                generated = output[:, inputs['input_ids'].shape[1]:]
                text = processor.tokenizer.decode(generated[0], skip_special_tokens=True)
                points = parse_points(text, transition.image.shape[1], transition.image.shape[0])
                point = points[0] if points else None
                scene_row['queries'][kind] = {
                    'text': text,
                    'points': points,
                    'metrics': point_metrics(point, targets[kind]),
                    'geometry_metrics': geometry_metrics(point, references[kind],
                                                         transition.image.shape[1],
                                                         transition.image.shape[0]),
                }
            rows.append(scene_row)
            if len(rows) <= 8:
                Image.fromarray(transition.image).save(args.output / f'{scene.scene_id}.png')
    finally:
        env.close()
    summaries = {}
    for kind in ('cube', 'target'):
        metrics = [row['queries'][kind]['metrics'] for row in rows]
        geometry = [row['queries'][kind]['geometry_metrics'] for row in rows]
        parsed = [item for item in metrics if item['parsed']]
        geometry_parsed = [item for item in geometry if item['parsed']]
        summaries[kind] = {
            'scenes': len(metrics),
            'parsed': len(parsed),
            'parse_rate': len(parsed) / len(metrics) if metrics else None,
            'inside_bbox': sum(item['inside_bbox'] for item in parsed),
            'inside_bbox_rate_parsed': (sum(item['inside_bbox'] for item in parsed) / len(parsed)
                                        if parsed else None),
            'mean_error_pixels_parsed': (float(np.mean([item['error_pixels'] for item in parsed]))
                                         if parsed else None),
            'mean_error_normalized_parsed': (float(np.mean([item['error_normalized'] for item in parsed]))
                                             if parsed else None),
            'geometry_reference': 'simulator projected center at z=0.02',
            'geometry_parsed': len(geometry_parsed),
            'geometry_within_32px': sum(item['within_32px'] for item in geometry_parsed),
            'geometry_within_32px_rate_parsed': (sum(item['within_32px'] for item in geometry_parsed) /
                                                 len(geometry_parsed) if geometry_parsed else None),
            'geometry_mean_error_pixels_parsed': (float(np.mean([item['error_pixels'] for item in geometry_parsed]))
                                                  if geometry_parsed else None),
            'geometry_mean_error_normalized_parsed': (
                float(np.mean([item['error_normalized'] for item in geometry_parsed]))
                if geometry_parsed else None),
        }
    report = {
        'config': {'model_id': args.model_id, 'revision': args.model_revision,
                   'scene_profile': 'directional_long', 'dev_scenes': args.dev_scenes,
                   'seed': args.seed, 'query': 'Molmo2 point grounding'},
        'summary': summaries,
        'scenes': rows,
        'rejected': rejected,
        'seconds': time.perf_counter() - start,
        'limitations': [
            'Color masks provide visible-image reference centers, not simulator projected centers.',
            'A parsed point outside the visible mask can still identify the object approximately.',
            'Pointing probe does not establish action-token or closed-loop control quality.',
        ],
    }
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'summary': summaries, 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    main()
