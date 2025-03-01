import torch
import torchaudio
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from typing import Optional, Tuple, List
import time

def load_models():
    draft_model = MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-small",
    )
    
    # Load the target (large) model
    target_model = draft_model
    # Load the processor (can use the same for both models)
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    
    return draft_model, target_model, processor

def generate_speculative(
    prompt: str,
    draft_model: MusicgenForConditionalGeneration,
    target_model: MusicgenForConditionalGeneration,
    processor: AutoProcessor,
    max_length: int = 6*64,
    look_ahead: int = 5,
) -> torch.Tensor:
    batch_size = 2
    
    encoder_inputs = processor(text=[prompt, prompt], padding=True, return_tensors="pt")
    encoder_hidden_states = None
    
    # (bsz, codebooks, sequence_length) 
    current_tokens = torch.ones((batch_size, 4, 1), dtype=torch.int32)
    
    # Generate until we reach max_length
    while current_tokens.shape[2] < max_length:
        for i in range(look_ahead):
            outputs = draft_model.forward(
                **encoder_inputs,
                encoder_hidden_states=encoder_hidden_states,
                decoder_input_ids=current_tokens,
                output_hidden_states=True,
            )
            
            # * (bsz*codebooks, sequence_length, vocab_size)
            logits = outputs.logits[:, -1:, :]
            # logits = torch.softmax(logits, dim=-1)
            next_token = torch.argmax(logits, dim=-1)# [-1:]
            next_token = next_token.unsqueeze(1)
            
            next_token = next_token.reshape((batch_size, 4, -1))
            
            current_tokens = torch.cat([current_tokens, next_token], dim=2)
       
        
    output_tokens = current_tokens
    # audio_scales = model_kwargs.get("audio_scales")
    # if audio_scales is None:
    audio_scales = [None] * batch_size

    # (1, bsz, codblocks, sequence_length)
    output_tokens = output_tokens.unsqueeze(0)
    output_values = draft_model.audio_encoder.decode(
        output_tokens,
        audio_scales=audio_scales,
    ).audio_values
    
    return output_values 
    


if __name__ == "__main__":
    # Example usage
    draft_model, target_model, processor = load_models()
    
    prompt = "Generate a happy electronic melody"
    
    start_time = time.time()
    with torch.no_grad():
        waveform = generate_speculative(
            prompt=prompt,
            draft_model=draft_model,
            target_model=target_model,
            processor=processor,
        )
    torchaudio.save("generated_music.mp3", waveform[0], draft_model.config.audio_encoder.sampling_rate)
    
    end_time = time.time()
    
    print(f"Generation took {end_time - start_time:.2f} seconds")
    