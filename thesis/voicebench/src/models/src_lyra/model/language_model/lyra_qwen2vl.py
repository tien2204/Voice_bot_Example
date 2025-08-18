
















from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2ForCausalLM, Qwen2VLModel
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from transformers.utils import logging
from transformers.generation.utils import GenerateOutput


from src.models.src_lyra.model.lyra_arch_qwen2vl import LyraMetaModel, LyraMetaForCausalLM
from torch.nn import CrossEntropyLoss


logger = logging.get_logger(__name__)

local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class LyraCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


class LyraConfig(Qwen2VLConfig):
    model_type = "Lyra_Qwen2VL"

class LyraQwen2VLModel(LyraMetaModel, Qwen2VLModel):
    config_class = LyraConfig
    
    def __init__(self, config: Qwen2VLConfig):
        super(LyraQwen2VLModel, self).__init__(config)


class LyraQwen2VLForCausalLM(Qwen2ForCausalLM, LyraMetaForCausalLM):
    config_class = LyraConfig

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = LyraQwen2VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        speeches: Optional[torch.FloatTensor] = None,
        speeches_asr: Optional[torch.FloatTensor] = None,
        audios: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        cache_position = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                loss_align,
                rope_deltas
            ) = self.prepare_inputs_labels_for_text_image_speech_qwen2vl(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                speeches,
                speeches_asr,
                rope_deltas
            )
            
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        loss = None
        
        if labels is not None:
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            
        if getattr(self.config, 'asr_align', None) and self.training:
            if self.device.index == 0:
                if isinstance(loss_align, float):
                    print("ASR Align loss: {:.4f}, LLM CE loss: {:.4f}".format(loss_align, loss.item()))
                else:
                    print("ASR Align loss: {:.4f}, LLM CE loss: {:.4f}".format(loss_align.item(), loss.item()))
            loss += self.config.weight_lambda * loss_align

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LyraCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        speeches: Optional[torch.Tensor] = None,
        speeches_asr: Optional[torch.Tensor] = None,
        audios: Optional[torch.Tensor] = None,
        rope_deltas: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None or speeches is not None or audios is not None:            
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _,
                rope_deltas
            ) = self.prepare_inputs_labels_for_text_image_speech_qwen2vl(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                speeches,
                speeches_asr,
                rope_deltas
            )
            
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            rope_deltas=rope_deltas,
            **kwargs
        )
    
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens,
        )

        if getattr(outputs, "rope_deltas", None) is not None:
            model_kwargs["rope_deltas"] = outputs.rope_deltas

        return model_kwargs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        speeches = kwargs.pop("speeches", None)
        speeches_asr = kwargs.pop("speeches_asr", None)
        audios = kwargs.pop("audios", None)
        rope_deltas = kwargs.pop("rope_deltas", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        if speeches is not None:
            _inputs['speeches'] = speeches
        if speeches_asr is not None:
            _inputs['speeches_asr'] = speeches_asr
        if audios is not None:
            _inputs['audios'] = audios
        if rope_deltas is not None:
            _inputs['rope_deltas'] = rope_deltas
        return _inputs

AutoConfig.register("Lyra_Qwen2VL", LyraConfig)
AutoModelForCausalLM.register(LyraConfig, LyraQwen2VLForCausalLM)
