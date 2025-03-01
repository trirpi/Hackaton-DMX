import copy
import torch
import torchaudio
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from typing import Optional, Tuple, List
import time
from transformers.generation import (
    ClassifierFreeGuidanceLogitsProcessor,
    GenerationConfig,
    GenerationMixin,
    GenerationMode,
    LogitsProcessorList,
    StoppingCriteriaList,
)

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    ModelOutput,
    Seq2SeqLMOutput,
)

def load_models():
    draft_model = MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-small",
    ).eval()  # Add eval() mode for inference
    
    # Load the target (large) model
    target_model = draft_model
    # Load the processor (can use the same for both models)
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    
    return draft_model, target_model, processor

def generate_speculative(
    inputs: str = None,
    draft_model: MusicgenForConditionalGeneration = None,
    target_model: MusicgenForConditionalGeneration = None,
    processor: AutoProcessor = None,
    max_length: int = 32,
    look_ahead: int = 1,
    **kwargs
) -> torch.Tensor:
    
    generation_config = draft_model.generation_config

    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
    generation_config.validate()
    draft_model._validate_model_kwargs(model_kwargs.copy())

    if model_kwargs.get("encoder_outputs") is not None and type(model_kwargs["encoder_outputs"]) is tuple:
        # wrap the unconditional outputs as a BaseModelOutput for compatibility with the rest of generate
        model_kwargs["encoder_outputs"] = BaseModelOutput(last_hidden_state=model_kwargs["encoder_outputs"][0])

    # 2. Set generation parameters if not already defined
    logits_processor = LogitsProcessorList()

    requires_attention_mask = "encoder_outputs" not in model_kwargs
    kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

    # 3. Define model inputs
    inputs_tensor, model_input_name, model_kwargs = draft_model._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    batch_size = inputs_tensor.shape[0]
    draft_model._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=inputs_tensor.device)

    # 4. Define other model kwargs
    model_kwargs["use_cache"] = False # generation_config.use_cache
    model_kwargs["guidance_scale"] = generation_config.guidance_scale

    if model_kwargs.get("attention_mask", None) is None and requires_attention_mask:
        model_kwargs["attention_mask"] = draft_model._prepare_attention_mask_for_generation(
            inputs_tensor, generation_config, model_kwargs
        )

    if "encoder_outputs" not in model_kwargs:
        # encoder_outputs are created and added to `model_kwargs`
        model_kwargs = draft_model._prepare_text_encoder_kwargs_for_generation(
            inputs_tensor, model_kwargs, model_input_name, generation_config
        )

    if "decoder_input_ids" not in model_kwargs and "input_values" in model_kwargs:
        model_kwargs = draft_model._prepare_audio_encoder_kwargs_for_generation(
            model_kwargs["input_values"],
            model_kwargs,
        )

    # 5. Prepare `input_ids` which will be used for auto-regressive generation
    input_ids, model_kwargs = draft_model._prepare_decoder_input_ids_for_generation(
        batch_size=batch_size,
        model_input_name=model_input_name,
        model_kwargs=model_kwargs,
        decoder_start_token_id=generation_config._decoder_start_token_tensor,
        bos_token_id=generation_config._bos_token_tensor,
        device=inputs_tensor.device,
    )

    # 6. Prepare `max_length` depending on other stopping criteria.
    input_ids_length = input_ids.shape[-1]
    has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
    generation_config = draft_model._prepare_generated_length(
        generation_config=generation_config,
        has_default_max_length=has_default_max_length,
        has_default_min_length=has_default_min_length,
        model_input_name=model_input_name,
        inputs_tensor=inputs_tensor,
        input_ids_length=input_ids_length,
    )

    # build the delay pattern mask for offsetting each codebook prediction by 1 (this behaviour is specific to MusicGen)
    input_ids, decoder_delay_pattern_mask = draft_model.decoder.build_delay_pattern_mask(
        input_ids,
        pad_token_id=generation_config._decoder_start_token_tensor,
        max_length=generation_config.max_length,
    )
    # stash the delay mask so that we don't have to recompute in each forward pass
    model_kwargs["decoder_delay_pattern_mask"] = decoder_delay_pattern_mask

    # 7. determine generation mode
    generation_mode = generation_config.get_generation_mode()

    # 8. prepare batched CFG externally (to enable coexistance with the unbatched CFG)
    if generation_config.guidance_scale is not None and generation_config.guidance_scale > 1:
        logits_processor.append(ClassifierFreeGuidanceLogitsProcessor(generation_config.guidance_scale))
        generation_config.guidance_scale = None

    # 9. prepare distribution pre_processing samplers
    logits_processor = draft_model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=None,
        logits_processor=logits_processor,
        device=input_ids.device,
    )

    # expand input_ids with `num_return_sequences` additional sequences per batch
    input_ids, model_kwargs = draft_model._expand_inputs_for_generation(
        input_ids=input_ids,
        expand_size=generation_config.num_return_sequences,
        is_encoder_decoder=draft_model.config.is_encoder_decoder,
        **model_kwargs,
    )
             
    
    
    current_tokens = input_ids #.unsqueeze(0).reshape((batch_size, 4, -1))
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    do_sample = generation_config.do_sample
    
    # Generate until we reach max_length
    while current_tokens.shape[1] < max_length:
        # model_kwargs["attention_mask"] = model_kwargs["attention_mask"].reshape((4, 1, 1, -1))
        for i in range(look_ahead):
            model_kwargs = draft_model._get_initial_cache_position(input_ids, model_kwargs)
            # prepare model inputs
            model_inputs = draft_model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            outputs = draft_model.forward(
                **model_inputs,
                # input_ids=input_ids,
                # decoder_input_ids=current_tokens,
                # output_hidden_states=True,
                # **model_kwargs
            )
            
            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = draft_model._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=draft_model.config.is_encoder_decoder,
            )
            
            # * (bsz*codebooks, sequence_length, vocab_size)
            logits = outputs.logits[:, -1:, :]
            # logits = torch.softmax(logits, dim=-1)
            
            next_token_scores = logits_processor(input_ids, logits)
            next_token = torch.argmax(next_token_scores, dim=-1)# [-1:]
            # next_token = next_token.reshape((batch_size, 4, -1))
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
       
        
    output_ids = current_tokens.reshape((batch_size*4, -1))
    output_ids = draft_model.decoder.apply_delay_pattern_mask(output_ids, decoder_delay_pattern_mask)
    output_ids = output_ids[output_ids != generation_config._pad_token_tensor].reshape(
        batch_size, draft_model.decoder.num_codebooks, -1
    )
    # audio_scales = model_kwargs.get("audio_scales")
    # if audio_scales is None:
    audio_scales = [None] * batch_size

    # (1, bsz, codblocks, sequence_length)
    output_ids = output_ids.unsqueeze(0)
    output_values = draft_model.audio_encoder.decode(
        output_ids,
        audio_scales=audio_scales,
    ).audio_values
    
    return output_values 
    


if __name__ == "__main__":
    # Example usage
    draft_model, target_model, processor = load_models()
    
    prompts = ["Generate a happy electronic melody"] * 2
    inputs = processor(text=prompts, padding=True, return_tensors="pt")
    
    start_time = time.time()
    with torch.no_grad():
        waveform = generate_speculative(
            **inputs,
            draft_model=draft_model,
            target_model=target_model,
            processor=processor,
        )
    torchaudio.save("generated_music.wav", waveform[0], draft_model.config.audio_encoder.sampling_rate)
    
    end_time = time.time()
    
    print(f"Generation took {end_time - start_time:.2f} seconds")
    