import sys, os, yaml, numpy as np, traceback
sys.argv += ['platform_hint', 'aloha']
sys.path[:0] = ['.', './description/utils', './policy/openvla-oft']
from envs import CONFIGS_PATH
from envs.lift_pot import lift_pot
from deploy_policy import Model, InferenceConfig, encode_obs

def make_args(name):
    with open(f'task_config/{name}.yml') as f: a = yaml.safe_load(f)
    a.update(task_name='lift_pot', task_config=name, ckpt_setting='model_contract_probe', eval_mode=True, eval_video_log=False, render_freq=0)
    with open(CONFIGS_PATH + '_embodiment_config.yml') as f: e = yaml.safe_load(f)
    es = a['embodiment']; lp = e[es[0]]['file_path']; rp = lp if len(es) == 1 else e[es[1]]['file_path']
    a['dual_arm_embodied'] = len(es) == 1
    if len(es) == 3: a['embodiment_dis'] = es[2]
    for side, path in [('left', lp), ('right', rp)]:
        a[f'{side}_robot_file'] = path
        with open(os.path.join(path, 'config.yml')) as f: a[f'{side}_embodiment_config'] = yaml.safe_load(f)
    with open(CONFIGS_PATH + '_camera_config.yml') as f: c = yaml.safe_load(f)
    a['head_camera_h'] = c['D435']['h']; a['head_camera_w'] = c['D435']['w']
    return a

ck = '/public/home/s2250018/wsy/VLA/checkpoints/lift_pot_sft_eval_l40'
print('LOAD_BEGIN', flush=True)
m = Model(InferenceConfig(pretrained_checkpoint=ck, use_l1_regression=False, use_diffusion=False, use_film=False,
                          use_proprio=True, num_images_in_input=1, center_crop=True, unnorm_key='lift_pot_1k',
                          num_open_loop_steps=25, lora_rank=32))
print('MODEL_OK', flush=True)
for name in ['demo_randomized', 'demo_randomized_aloha']:
    t = lift_pot()
    try:
        t.setup_demo(now_ep_num=0, seed=100000, is_test=True, **make_args(name))
        raw = t.get_obs(); raw['language'] = 'lift the pot'; enc = encode_obs(raw); act = np.asarray(m.get_action(raw))
        print('CONTRACT', name, 'state', enc['state'].shape, 'img', enc['full_image'].shape, 'actions', act.shape,
              'finite', bool(np.isfinite(act).all()), 'minmax', float(act.min()), float(act.max()), flush=True)
    except Exception as e:
        print('CONTRACT_FAIL', name, type(e).__name__, str(e), flush=True); traceback.print_exc(limit=3)
    finally:
        try: t.close_env(clear_cache=True)
        except Exception: pass
print('PROBE_DONE', flush=True)
