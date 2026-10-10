from __future__ import annotations
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import random
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from .actions import ACTION_LABELS
from .env_adapter import Transition
from .feedback import FeedbackRecord, build_feedback_text

MODEL_ID = 'Qwen/Qwen2.5-VL-3B-Instruct'
MODEL_REVISION = '66285546d2b821cf421d4f5eb2576359d3770cd3'
ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PolicyConfig:
    model_id: str = str(ROOT / 'models/Qwen2.5-VL-3B-Instruct')
    revision: str = MODEL_REVISION
    min_pixels: int = 448 * 448
    max_pixels: int = 448 * 448
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    teacher_update_rate: float = 0.05
    rollout_temperature: float = 1.0
    distillation_loss: str = 'js'
    seed: int = 11


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class QwenActionPolicy:
    def __init__(self, config: PolicyConfig = PolicyConfig(), device='cuda'):
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; activate sdpo-vla-pilot before loading policy')
        seed_everything(config.seed)
        self.config, self.device = config, torch.device(device)
        local = Path(config.model_id).is_dir()
        options = {'local_files_only': True} if local else {'revision':config.revision}
        self.processor = AutoProcessor.from_pretrained(config.model_id,
            min_pixels=config.min_pixels,max_pixels=config.max_pixels,**options)
        self.processor.tokenizer.padding_side = 'left'
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config.model_id,
            torch_dtype=torch.bfloat16, device_map={'':str(self.device)},
            attn_implementation='sdpa', **options)
        targets = [name for name,module in model.named_modules()
                   if name.startswith('model.language_model.layers.')
                   and name.rsplit('.',1)[-1] in {'q_proj','k_proj','v_proj','o_proj'}
                   and isinstance(module,torch.nn.Linear)]
        if not targets:
            raise RuntimeError('No language-only attention LoRA target modules found')
        self.model = get_peft_model(model,LoraConfig(r=config.lora_rank,lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,target_modules=targets,task_type='CAUSAL_LM'))
        self.model.eval()
        unexpected = [name for name,p in self.model.named_parameters() if p.requires_grad
                      and ('.language_model.layers.' not in name or 'lora_' not in name)]
        if unexpected:
            raise RuntimeError(f'Non-language LoRA parameters are trainable: {unexpected}')
        self.action_token_ids = self._resolve_action_tokens()
        self.optimizer = torch.optim.AdamW(self.trainable_parameters(),lr=2e-5,weight_decay=0.)
        self.teacher_ema = {name: parameter.detach().float().cpu().clone()
            for name,parameter in self.model.named_parameters() if parameter.requires_grad}
        self.sampler = torch.Generator(device=self.device).manual_seed(config.seed)
        model_id = f'Qwen/{Path(config.model_id).name}' if local else config.model_id
        self.audit = {'model_id':model_id,'revision':config.revision,'local_path':config.model_id,
            'action_token_ids':dict(zip(ACTION_LABELS,self.action_token_ids)),
            'trainable_parameters':sum(p.numel() for p in self.trainable_parameters()),
            'lora_targets':targets,'image_pixels':config.max_pixels,'padding_side':'left',
            'attention':'sdpa','teacher':'separate_lora_ema',
            'teacher_update_rate':config.teacher_update_rate,
            'rollout_temperature':config.rollout_temperature,
            'distillation':config.distillation_loss,
            'action_support':'five_action_full_distribution',
            'distillation_temperature':1.0}

    def trainable_parameters(self) -> Iterable[torch.nn.Parameter]:
        return (p for p in self.model.parameters() if p.requires_grad)

    def _resolve_action_tokens(self):
        tokenizer = self.processor.tokenizer
        text = self.processor.apply_chat_template(
            [{'role':'user','content':[{'type':'text','text':'Choose A, B, C, D or E.'}]}],
            tokenize=False,add_generation_prompt=True)
        prefix_ids = tokenizer.encode(text,add_special_tokens=False)
        ids = []
        for label in ACTION_LABELS:
            complete = tokenizer.encode(text+label,add_special_tokens=False)
            if complete[:-1] != prefix_ids or tokenizer.decode(complete[-1:]) != label:
                raise ValueError(f'Action {label} is not one token at the exact response boundary')
            ids.append(complete[-1])
        if len(set(ids)) != 5:
            raise ValueError('Action labels must be distinct')
        return ids

    @staticmethod
    def _prompt(transition: Transition, feedback=False, feedback_record=None,
                solution_action=None, include_outcome_feedback=True,
                additional_context=None):
        xyz = ', '.join(f'{v:.4f}' for v in transition.proprio)
        text = (f'You control a robot arm. Task: {transition.instruction}\n'
            'The solid green cube is the object; the translucent blue square is the target.\n'
            'The fixed lower-left arrow legend shows world +x and +y directions.\n'
            f'Current end-effector position in world coordinates (meters): [{xyz}].\n'
            'Actions: A=world +x by 0.02 m, B=world -x by 0.02 m, '
            'C=world +y by 0.02 m, D=world -y by 0.02 m, E=hold.\n'
            'The closed gripper stays at a fixed pushing height; each action lasts 0.16 seconds.\n'
            'Choose one action label for the current image/state.\n')
        if feedback:
            if solution_action is not None:
                text += ('\nCorrect self-generated solution for the SAME original image/state:\n\n'
                         f'{solution_action}\n\nCorrectly solve the original task using this self-generated demonstration.\n')
            if include_outcome_feedback:
                result = feedback_record if feedback_record is not None else transition.result
                if result is None:
                    raise ValueError('Teacher requires an explicitly reported feedback record')
                text += '\nRetrospective result feedback:\n' + build_feedback_text(transition.action_label,result)
            if solution_action is None and not include_outcome_feedback:
                raise ValueError('Teacher prompt needs a self demonstration or outcome feedback')
        if additional_context:
            text += f'\nAdditional diagnostic input:\n{additional_context}\n'
        return text + '\nReturn exactly one of A, B, C, D, E.'

    def _inputs(self, transitions, feedback, feedback_records=None, solution_actions=None,
                include_outcome_feedback=True, additional_context=None):
        if feedback_records is not None and len(feedback_records)!=len(transitions):
            raise ValueError('Feedback records must align one-to-one with transitions')
        if solution_actions is not None and len(solution_actions)!=len(transitions):
            raise ValueError('Solution actions must align one-to-one with transitions')
        texts=[]
        for i,t in enumerate(transitions):
            f = feedback_records[i] if feedback_records is not None else None
            solution = solution_actions[i] if solution_actions is not None else None
            context = (additional_context[i] if isinstance(additional_context, list)
                       else additional_context)
            include_feedback=(include_outcome_feedback[i] if isinstance(include_outcome_feedback,list)
                              else include_outcome_feedback)
            messages=[{'role':'user','content':[{'type':'image'},
                {'type':'text','text':self._prompt(t,feedback,f,solution,include_feedback,context)}]}]
            texts.append(self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True))
        images=[Image.fromarray(t.image).convert('RGB') for t in transitions]
        inputs=self.processor(text=texts,images=images,return_tensors='pt',padding=True)
        if not bool(inputs['attention_mask'][:,-1].all()):
            raise RuntimeError('The last position must contain a real token, not padding')
        return {k:v.to(self.device) if torch.is_tensor(v) else v for k,v in inputs.items()}

    def log_probs(self, transitions, feedback=False, grad=False, feedback_records=None,
                  solution_actions=None, include_outcome_feedback=True,
                  additional_context=None):
        if not transitions:
            raise ValueError('At least one transition is required')
        with nullcontext() if grad else torch.no_grad():
            inputs=self._inputs(transitions,feedback,feedback_records,solution_actions,
                                include_outcome_feedback,additional_context)
            # PEFT adapters are installed within the language decoder. Calling its
            # multimodal body keeps these adapters active; evaluate only the five
            # selected frozen LM-head rows in float32 (avoid BF16 logit rounding).
            base=self.model.get_base_model()
            outputs=base.model(**inputs,use_cache=False,return_dict=True)
            features=outputs.last_hidden_state[:,-1,:].float()
            weight=base.lm_head.weight[self.action_token_ids].float()
            logits=torch.nn.functional.linear(features,weight)
            return torch.log_softmax(logits,dim=-1)

    @contextmanager
    def _ema_teacher_weights(self):
        student = {}
        with torch.no_grad():
            for name,parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    student[name] = parameter.detach().clone()
                    parameter.copy_(self.teacher_ema[name].to(device=parameter.device,dtype=parameter.dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                for name,parameter in self.model.named_parameters():
                    if parameter.requires_grad:
                        parameter.copy_(student[name])

    def teacher_log_probs(self, transitions, feedback=False, feedback_records=None,
                          solution_actions=None, include_outcome_feedback=True):
        with self._ema_teacher_weights():
            return self.log_probs(transitions,feedback=feedback,grad=False,
                                  feedback_records=feedback_records,solution_actions=solution_actions,
                                  include_outcome_feedback=include_outcome_feedback)

    @torch.no_grad()
    def update_teacher_ema(self):
        rate=self.config.teacher_update_rate
        for name,parameter in self.model.named_parameters():
            if parameter.requires_grad:
                self.teacher_ema[name].mul_(1.-rate).add_(parameter.detach().float().cpu(),alpha=rate)

    @torch.no_grad()
    def sample_batch(self, transitions, greedy=False, additional_context=None):
        log_probs=self.log_probs(transitions,additional_context=additional_context)
        probs=(log_probs if greedy else log_probs/self.config.rollout_temperature).softmax(-1)
        indices=probs.argmax(-1) if greedy else torch.multinomial(probs,1,generator=self.sampler).squeeze(-1)
        return [ACTION_LABELS[i] for i in indices.cpu().tolist()]

    def sample(self,transition):
        return self.sample_batch([transition])[0]

    @torch.no_grad()
    def generate_reflection(self, transition):
        """Generate a bounded retrospective suggestion from one real outcome."""
        result = transition.result
        prompt = (f'You control a robot arm. Task: {transition.instruction}\n'
            'Actions: A=world +x, B=world -x, C=world +y, D=world -y, E=hold.\n'
            f'Attempted action: {transition.action_label}.\n'
            'Observed result at the original state:\n'
            f'Contact: {result.contact}. Cube motion: {result.object_motion}. '
            f'Target distance change: {result.goal_progress}. Success: {result.success}.\n\n'
            'Give one short correction for choosing the next action at the ORIGINAL state. '
            'State what went wrong and which action should be tried next.')
        messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}]
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.processor(text=[text],images=[Image.fromarray(transition.image).convert('RGB')],
                              return_tensors='pt',padding=True)
        inputs={k:v.to(self.device) if torch.is_tensor(v) else v for k,v in inputs.items()}
        output=self.model.generate(**inputs,max_new_tokens=64,do_sample=False,use_cache=True)
        prompt_length=inputs['input_ids'].shape[1]
        return self.processor.batch_decode(output[:,prompt_length:],skip_special_tokens=True)[0].strip()

    def reflection_feedback(self, transition):
        from dataclasses import replace
        return replace(transition.result,
                       interpretation=f'Model-generated reflection: {self.generate_reflection(transition)}')

    def save(self,path):
        path=Path(path)
        path.mkdir(parents=True,exist_ok=True)
        self.model.save_pretrained(path)
        # This local checkpoint includes optimizer state for an exact same-environment resume.
        torch.save({'optimizer':self.optimizer.state_dict(),'torch_rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all(),'sampler_rng':self.sampler.get_state(),
                    'teacher_ema':self.teacher_ema},path/'training_state.pt')
