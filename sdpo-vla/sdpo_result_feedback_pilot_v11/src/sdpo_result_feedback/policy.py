from __future__ import annotations
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable
import random
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
from .actions import ACTION_LABELS
from .env_adapter import Transition
from .feedback import FeedbackRecord, build_feedback_text

ROOT = Path(__file__).resolve().parents[2]

MODEL_PRESETS = {
    'qwen3': ROOT / 'models/Qwen3-VL-8B-Instruct',
    'internvl3.5': ROOT / 'models/InternVL3_5-8B-HF',
    'molmo2': ROOT / 'models/Molmo2-8B',
}

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


@dataclass(frozen=True)
class PolicyConfig:
    model_id: str = str(MODEL_PRESETS['qwen3'])
    revision: str | None = None
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


class ActionPolicy:
    def __init__(self, config: PolicyConfig = PolicyConfig(), device='cuda'):
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; activate sdpo-vla-pilot before loading policy')
        seed_everything(config.seed)
        self.config, self.device = config, torch.device(device)
        local = Path(config.model_id).is_dir()
        options = {'local_files_only': True} if local else ({'revision':config.revision}
            if config.revision else {})
        model_config = AutoConfig.from_pretrained(config.model_id, trust_remote_code=True, **options)
        try:
            self.processor = AutoProcessor.from_pretrained(config.model_id,
                min_pixels=config.min_pixels,max_pixels=config.max_pixels,
                trust_remote_code=True,**options)
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(config.model_id,
                trust_remote_code=True,**options)
        self.processor.tokenizer.padding_side = 'left'
        requested_attention = os.environ.get('SDPO_ATTN_IMPLEMENTATION', 'flash_attention_2')
        if requested_attention not in {'sdpa', 'flash_attention_2', 'eager'}:
            raise ValueError('SDPO_ATTN_IMPLEMENTATION must be sdpa, flash_attention_2, or eager')
        try:
            model = AutoModelForImageTextToText.from_pretrained(config.model_id,
                dtype=torch.bfloat16, device_map={'':str(self.device)},
                attn_implementation=requested_attention, trust_remote_code=True, **options)
        except (TypeError, ValueError) as error:
            if requested_attention == 'sdpa' or (
                    'attn_implementation' not in str(error)
                    and 'flash' not in str(error).lower()
                    and 'unexpected keyword' not in str(error)):
                raise
            model = AutoModelForImageTextToText.from_pretrained(config.model_id,
                dtype=torch.bfloat16, device_map={'':str(self.device)},
                attn_implementation='sdpa', trust_remote_code=True, **options)
            requested_attention = 'sdpa'
        targets = []
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            parts = name.split('.')
            is_language_attention = (
                ('.language_model.' in f'.{name}' and '.layers.' in name
                 and name.rsplit('.', 1)[-1] in {'q_proj', 'k_proj', 'v_proj', 'o_proj'})
                or ('.model.transformer.blocks.' in f'.{name}'
                    and name.rsplit('.', 1)[-1] in {'att_proj', 'attn_out'})
            )
            if is_language_attention and 'vision_backbone' not in parts:
                targets.append(name)
        if not targets:
            raise RuntimeError('No language-only attention LoRA target modules found')
        self.model = get_peft_model(model,LoraConfig(r=config.lora_rank,lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,target_modules=targets,task_type='CAUSAL_LM'))
        self._base_model = self.model.get_base_model()
        self._lm_head = getattr(self._base_model, 'lm_head', None)
        if self._lm_head is None or not hasattr(self._lm_head, 'weight'):
            raise RuntimeError('Loaded multimodal model has no top-level lm_head')
        self.model.eval()
        unexpected = [name for name,p in self.model.named_parameters() if p.requires_grad
                      and (('language_model' not in name and '.model.transformer.' not in name)
                           or 'lora_' not in name)]
        if unexpected:
            raise RuntimeError(f'Non-language LoRA parameters are trainable: {unexpected}')
        self.action_token_ids = self._resolve_action_tokens()
        self.optimizer = torch.optim.AdamW(self.trainable_parameters(),lr=2e-5,weight_decay=0.)
        self.teacher_ema = {name: parameter.detach().float().cpu().clone()
            for name,parameter in self.model.named_parameters() if parameter.requires_grad}
        self.sampler = torch.Generator(device=self.device).manual_seed(config.seed)
        self.audit = {'model_id':config.model_id,'revision':config.revision,'local_path':config.model_id,
            'model_type':getattr(model_config,'model_type',None),
            'architecture':model.__class__.__name__,
            'action_token_ids':dict(zip(ACTION_LABELS,self.action_token_ids)),
            'trainable_parameters':sum(p.numel() for p in self.trainable_parameters()),
            'lora_targets':targets,'image_pixels':config.max_pixels,'padding_side':'left',
            'attention':requested_attention,'teacher':'separate_lora_ema',
            'teacher_update_rate':config.teacher_update_rate,
            'rollout_temperature':config.rollout_temperature,
            'distillation':config.distillation_loss,
            'action_support':'five_action_full_distribution',
            'distillation_temperature':1.0}

    def trainable_parameters(self) -> Iterable[torch.nn.Parameter]:
        return (p for p in self.model.parameters() if p.requires_grad)

    def _resolve_action_tokens(self):
        tokenizer = self.processor.tokenizer
        text = self._chat_template(
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

    def _chat_template(self, messages, **kwargs):
        try:
            return self.processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except (TypeError, ValueError):
            return self.processor.apply_chat_template(messages, **kwargs)

    @staticmethod
    def _prompt(transition: Transition, feedback=False, feedback_record=None,
                solution_action=None, include_outcome_feedback=True,
                additional_context=None, action_prefix=None):
        xyz = ', '.join(f'{v:.4f}' for v in transition.proprio)
        text = (f'You control a robot arm. Task: {transition.instruction}\n'
            'The solid green cube is the object; the translucent blue square is the target.\n'
            'The fixed lower-left arrow legend shows world +x and +y directions.\n'
            f'Current end-effector position in world coordinates (meters): [{xyz}].\n'
            'Actions: A=world +x by 0.02 m, B=world -x by 0.02 m, '
            'C=world +y by 0.02 m, D=world -y by 0.02 m, E=hold.\n'
            'The closed gripper stays at a fixed pushing height; each action lasts 0.16 seconds.\n'
            'Choose one action label for the current image/state.\n')
        if action_prefix is not None:
            text += ('\nThis is one position in an autoregressive action chunk; '
                     'predict the next action token from the response prefix.\n')
        if feedback:
            if solution_action is not None:
                text += ('\nCorrect self-generated solution for the SAME original image/state:\n\n'
                         f'{solution_action}\n\nCorrectly solve the original task using this self-generated demonstration.\n')
            if include_outcome_feedback:
                result = feedback_record if feedback_record is not None else transition.result
                if result is None:
                    raise ValueError('Teacher requires an explicitly reported feedback record')
                action = (' '.join(transition.action_labels)
                          if hasattr(transition, 'action_labels') else transition.action_label)
                text += '\nRetrospective result feedback:\n' + build_feedback_text(action, result)
            if solution_action is None and not include_outcome_feedback:
                raise ValueError('Teacher prompt needs a self demonstration or outcome feedback')
        if additional_context:
            text += f'\nAdditional diagnostic input:\n{additional_context}\n'
        return text + '\nReturn exactly one of A, B, C, D, E.'

    def _inputs(self, transitions, feedback, feedback_records=None, solution_actions=None,
                include_outcome_feedback=True, additional_context=None,
                action_prefixes=None):
        if feedback_records is not None and len(feedback_records)!=len(transitions):
            raise ValueError('Feedback records must align one-to-one with transitions')
        if solution_actions is not None and len(solution_actions)!=len(transitions):
            raise ValueError('Solution actions must align one-to-one with transitions')
        if action_prefixes is not None and len(action_prefixes)!=len(transitions):
            raise ValueError('Action prefixes must align one-to-one with transitions')
        texts=[]
        for i,t in enumerate(transitions):
            f = feedback_records[i] if feedback_records is not None else None
            solution = solution_actions[i] if solution_actions is not None else None
            context = (additional_context[i] if isinstance(additional_context, list)
                       else additional_context)
            prefix = action_prefixes[i] if action_prefixes is not None else None
            include_feedback=(include_outcome_feedback[i] if isinstance(include_outcome_feedback,list)
                              else include_outcome_feedback)
            messages=[{'role':'user','content':[{'type':'image'},
                {'type':'text','text':self._prompt(t,feedback,f,solution,include_feedback,context,prefix)}]}]
            texts.append(self._chat_template(messages,tokenize=False,add_generation_prompt=True))
        images=[Image.fromarray(t.image).convert('RGB') for t in transitions]
        inputs=self.processor(text=texts,images=images,return_tensors='pt',padding=True)
        if not bool(inputs['attention_mask'][:,-1].all()):
            raise RuntimeError('The last position must contain a real token, not padding')
        return {k:v.to(self.device) if torch.is_tensor(v) else v for k,v in inputs.items()}

    def log_probs(self, transitions, feedback=False, grad=False, feedback_records=None,
                  solution_actions=None, include_outcome_feedback=True,
                  additional_context=None, action_prefixes=None):
        if not transitions:
            raise ValueError('At least one transition is required')
        with nullcontext() if grad else torch.no_grad():
            inputs=self._inputs(transitions,feedback,feedback_records,solution_actions,
                                include_outcome_feedback,additional_context,action_prefixes)
            if action_prefixes is not None:
                original_length = inputs['input_ids'].shape[1]
                prefix_ids = torch.as_tensor(
                    [[self.action_token_ids['ABCDE'.index(action)] for action in prefix]
                     for prefix in action_prefixes], dtype=torch.long, device=self.device)
                if prefix_ids.shape[1]:
                    inputs['input_ids'] = torch.cat([inputs['input_ids'], prefix_ids], dim=1)
                    inputs['attention_mask'] = torch.cat([
                        inputs['attention_mask'],
                        torch.ones((len(transitions), prefix_ids.shape[1]),
                                   dtype=inputs['attention_mask'].dtype, device=self.device),
                    ], dim=1)
                    if 'position_ids' in inputs:
                        start = inputs['position_ids'][:, -1:] + 1
                        offsets = torch.arange(prefix_ids.shape[1], device=self.device).view(1, -1)
                        inputs['position_ids'] = torch.cat([
                            inputs['position_ids'], start + offsets], dim=1)
                    if ('token_type_ids' in inputs
                            and inputs['token_type_ids'].shape[1] == original_length):
                        tail = inputs['token_type_ids'][:, -1:].expand(
                            -1, prefix_ids.shape[1])
                        inputs['token_type_ids'] = torch.cat(
                            [inputs['token_type_ids'], tail], dim=1)
            # PEFT adapters are installed within the language decoder. Calling its
            # multimodal body keeps these adapters active; evaluate only the five
            # selected frozen LM-head rows in float32 (avoid BF16 logit rounding).
            base=self._base_model
            body=getattr(base,'model',None)
            if body is None:
                raise RuntimeError('Loaded multimodal model has no model body')
            outputs=body(**inputs,use_cache=False,return_dict=True)
            features=outputs.last_hidden_state[:,-1,:].float()
            weight=self._lm_head.weight[self.action_token_ids].float()
            logits=torch.nn.functional.linear(features,weight)
            return torch.log_softmax(logits,dim=-1)

    def chunk_log_probs(self, transitions, action_chunks, feedback=False,
                        grad=False, feedback_records=None, solution_actions=None,
                        include_outcome_feedback=True, batch_size=None,
                        feedback_records_by_position=None):
        """Return [batch, chunk_length, action_count] with causal prefixes."""
        if not transitions or len(action_chunks) != len(transitions):
            raise ValueError('Chunk inputs must align with transitions')
        lengths = {len(chunk) for chunk in action_chunks}
        if len(lengths) != 1 or not next(iter(lengths)):
            raise ValueError('All action chunks must have the same positive length')
        chunk_length = next(iter(lengths))
        if feedback_records_by_position is not None:
            if len(feedback_records_by_position) != chunk_length:
                raise ValueError('Position feedback must match chunk length')
            if any(len(records) != len(transitions)
                   for records in feedback_records_by_position):
                raise ValueError('Position feedback must align with transitions')
        batch_size = batch_size or len(transitions)
        rows = []
        for position in range(chunk_length):
            prefixes = [tuple(chunk[:position]) for chunk in action_chunks]
            position_solutions = [
                solution[position] if solution is not None else None
                for solution in solution_actions
            ] if solution_actions is not None else None
            batches = []
            for start in range(0, len(transitions), batch_size):
                end = start + batch_size
                position_records = (
                    feedback_records_by_position[position][start:end]
                    if feedback_records_by_position is not None else
                    feedback_records[start:end] if feedback_records is not None else None
                )
                batches.append(self.log_probs(
                    transitions[start:end], feedback=feedback, grad=grad,
                    feedback_records=position_records,
                    solution_actions=position_solutions[start:end]
                    if position_solutions is not None else None,
                    include_outcome_feedback=include_outcome_feedback[start:end]
                    if isinstance(include_outcome_feedback, list)
                    else include_outcome_feedback,
                    action_prefixes=prefixes[start:end]))
            rows.append(torch.cat(batches, dim=0))
        return torch.stack(rows, dim=1)

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
                          solution_actions=None, include_outcome_feedback=True,
                          additional_context=None):
        with self._ema_teacher_weights():
            return self.log_probs(transitions,feedback=feedback,grad=False,
                                  feedback_records=feedback_records,solution_actions=solution_actions,
                                  include_outcome_feedback=include_outcome_feedback,
                                  additional_context=additional_context)

    def teacher_chunk_log_probs(self, transitions, action_chunks, feedback=False,
                                feedback_records=None, solution_actions=None,
                                include_outcome_feedback=True, batch_size=None,
                                feedback_records_by_position=None):
        with self._ema_teacher_weights():
            return self.chunk_log_probs(
                transitions, action_chunks, feedback=feedback, grad=False,
                feedback_records=feedback_records,
                feedback_records_by_position=feedback_records_by_position,
                solution_actions=solution_actions,
                include_outcome_feedback=include_outcome_feedback,
                batch_size=batch_size)

    @torch.no_grad()
    def update_teacher_ema(self):
        rate=self.config.teacher_update_rate
        for name,parameter in self.model.named_parameters():
            if parameter.requires_grad:
                self.teacher_ema[name].mul_(1.-rate).add_(parameter.detach().float().cpu(),alpha=rate)

    @torch.no_grad()
    def sample_batch(self, transitions, greedy=False, additional_context=None):
        repeated_state = len(transitions) > 1 and all(
            transition is transitions[0] for transition in transitions)
        scoring_transitions = transitions[:1] if repeated_state else transitions
        log_probs=self.log_probs(scoring_transitions,additional_context=additional_context)
        probs=(log_probs if greedy else log_probs/self.config.rollout_temperature).softmax(-1)
        if repeated_state:
            probs=probs.expand(len(transitions), -1)
        indices=probs.argmax(-1) if greedy else torch.multinomial(probs,1,generator=self.sampler).squeeze(-1)
        return [ACTION_LABELS[i] for i in indices.cpu().tolist()]

    @torch.no_grad()
    def sample_chunk(self, transitions, chunk_length, greedy=False, batch_size=None):
        if chunk_length < 1:
            raise ValueError('Chunk length must be positive')
        prefixes = [[] for _ in transitions]
        batch_size = batch_size or len(transitions)
        for _ in range(chunk_length):
            for start in range(0, len(transitions), batch_size):
                end = start + batch_size
                log_probs = self.log_probs(
                    transitions[start:end],
                    action_prefixes=[tuple(prefix) for prefix in prefixes[start:end]])
                probs = (log_probs if greedy else
                         log_probs / self.config.rollout_temperature).softmax(-1)
                indices = (probs.argmax(-1) if greedy else
                           torch.multinomial(probs, 1, generator=self.sampler).squeeze(-1))
                for prefix, index in zip(prefixes[start:end], indices.cpu().tolist()):
                    prefix.append(ACTION_LABELS[index])
        return [tuple(prefix) for prefix in prefixes]

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
        text=self._chat_template(messages,tokenize=False,add_generation_prompt=True)
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


QwenActionPolicy = ActionPolicy
