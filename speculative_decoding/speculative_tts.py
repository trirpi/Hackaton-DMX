import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

import outetts
import torch
import torchaudio
import torchaudio.transforms as transforms
from fastdtw import fastdtw
from scipy.spatial.distance import cosine, euclidean
from transformers import (
    AutoTokenizer,
    VitsModel,
    pipeline,
    set_seed,
)

MAX_CHUNK_LEN = 10
SPECULATION_BATCH_SIZE = 6
VERIFICATION_SWITCH = 1
MELSPECTRAM_THRESHOLD = 0.9
ASR_THRESHOLD = 0.06 # this is chosen low because VitsModel and ASR model are very lousy

#############################################
# 1) Draft TTS: small, fast text-to-speech
#############################################
class DraftTTS:
    """
    A small TTS model for quick, lower-quality generation.
    Example: facebook/mms-tts-eng (~300M params).
    """

    def __init__(self, model_name="facebook/mms-tts-eng", device="cuda"):
        self.device = device
        print(f"[DraftTTS] Loading model: {model_name}")
        # MMS TTS (English)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = VitsModel.from_pretrained(model_name).to(device).eval()
        self.sample_rate = 16000

    def generate(self, text: str, seed: int = 42) -> torch.Tensor:
        """
        Generate a single-audio chunk from text. Returns waveform as a 1D float tensor (CPU).
        """
        set_seed(seed)
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            output = self.model(**inputs)
            waveform = output.waveform[0]
        return waveform.cpu()  # Move to CPU for easy handling


#############################################
# 2) Main TTS: larger, higher-quality model
#############################################
class MainTTS:
    """
    A bigger TTS for higher-quality fallback.
    Example: OuteAI/OuteTTS-0.3-1B (~1B params).
    """

    def __init__(self, model_name="OuteAI/OuteTTS-0.3-1B"):
        print(f"[MainTTS] Loading model: {model_name}")
        model_config = outetts.HFModelConfig_v2(
            model_path=model_name,
            tokenizer_path=model_name,
        )
        self.model = outetts.InterfaceHF(model_version="0.3", cfg=model_config)
        self.model.print_default_speakers()
        # Load default speaker
        self.speaker = self.model.load_default_speaker(name="en_male_1")

        # Generation config
        self.gen_cfg = outetts.GenerationConfig(
            text=None,
            temperature=0.4,
            repetition_penalty=1.1,
            max_length=4096,
            speaker=self.speaker,
        )
        self.sample_rate = 48000
        self.prompt_processor = self.model.prompt_processor

    def generate(self, text: str, seed: int = 42) -> torch.Tensor:
        set_seed(seed)
        self.gen_cfg.text = text
        with torch.no_grad():
            output = self.model.generate(config=self.gen_cfg)
        return output.audio.squeeze(0).cpu()


#############################################
# 3) ASR Verifier
#############################################
class ASRVerifier:
    """
    Uses a Whisper-based (or other) speech recognition model to confirm
    the TTS audio matches the reference text.
    """

    def __init__(self, model_name="openai/whisper-base", main_tts=None, device="cuda"):
        print(f"[ASRVerifier] Loading ASR model: {model_name}")
        self.device = 0 if device == "cuda" else -1
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=self.device,
        )
        self.asr_sample_rate = 16000
        self.acceptance_threshold = ASR_THRESHOLD
        self.main_tts = main_tts
        
    # verify_v1 relies on ASR to verify the generated audio
    def verify_v1(self, text: str, audio_waveform: torch.Tensor, original_sr: int) -> bool:
        # Convert waveform to numpy
        audio_waveform_np = audio_waveform.numpy()

        # Transcribe (Fix: Remove 'sampling_rate' from kwargs)
        recognized = self.asr({"array": audio_waveform_np, "sampling_rate": self.asr_sample_rate})["text"]
        recognized = recognized.strip().lower()
        reference = text.strip().lower()

        # Similarity score
        ratio = SequenceMatcher(None, reference, recognized).ratio()
        print(f"Accept draft model? ", ratio >= self.acceptance_threshold)
        return ratio >= self.acceptance_threshold

    # verify_v2 relies on MelSpectrogram similarity to verify the generated audio, requiring full iteration of the main model execution
    def verify_v2(self, text: str, draft_audio_waveform: torch.Tensor, original_sr: int) -> bool:
        if self.main_tts:
            self.prompt_processor = self.main_tts.prompt_processor

        print("[verify_v2] Generating reference audio tokens from Main TTS...")
        main_audio_waveform = self.main_tts.generate(text)  # Get waveform from Main TTS
        
        alignment_score = self._spectrogram_similarity(main_audio_waveform, draft_audio_waveform)
        if alignment_score > MELSPECTRAM_THRESHOLD:
            return True
        return False

    def _spectrogram_similarity(self, waveform1: torch.Tensor, waveform2: torch.Tensor, sample_rate=48000) -> float:
        """
        Compute similarity between spectrogram representations using cosine similarity.

        Returns:
            A similarity score in [0,1], where 1.0 means perfect match.
        """
        # Convert waveform to spectrogram
        spectrogram_transform = transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=128)
        spec1 = spectrogram_transform(waveform1.unsqueeze(0)).mean(dim=-1).squeeze()
        spec2 = spectrogram_transform(waveform2.unsqueeze(0)).mean(dim=-1).squeeze()

        # Normalize to unit vectors
        spec1 = spec1 / torch.norm(spec1)
        spec2 = spec2 / torch.norm(spec2)

        # Compute cosine similarity (higher = more similar)
        similarity = 1 - cosine(spec1.numpy(), spec2.numpy())

        return max(0, min(similarity, 1))  # Ensure score is in [0,1]


#############################################
# 4) Speculative TTS Pipeline
#############################################
class SpeculativeTTSPipeline:
    def __init__(
        self,
        draft_model: DraftTTS,
        main_model: MainTTS,
        verifier: ASRVerifier,
        speculation_batch_size: int = 4,
        crossfade_sec: float = 0.05,
        device="cuda",
    ):
        self.draft = draft_model
        self.main = main_model
        self.verifier = verifier
        self.verifier.main_tts = self.main
        self.speculation_batch_size = speculation_batch_size
        self.crossfade_sec = crossfade_sec
        self.device = device

        # We'll unify on the draft sample_rate for final output, though both are 48k anyway
        self.sample_rate = self.main.sample_rate
        self.max_workers = 4 if device == "cuda" else 2

    def generate_speech(self, text: str, seed: int = 42) -> torch.Tensor:
        """
        High-level method to produce final audio from text in a "speculative" manner:
         - Batch up chunks
         - Generate them with the draft model in parallel
         - Verify each chunk with ASR
         - If rejected, re-generate chunk with main model
         - Crossfade the final pieces
        """
        set_seed(seed)
        chunks = self._split_text(text)
        print(f"[SpeculativeTTS] {len(chunks)} text chunk(s) identified.")

        final_segments = []
        idx = 0

        while idx < len(chunks):
            # pick up to speculation_batch_size chunks
            batch_chunks = chunks[idx : idx + self.speculation_batch_size]

            # 1) Draft generation in parallel
            waveforms = self._draft_generate_batch(batch_chunks, seed=seed)

            # Resample to 48kHz if necessary
            new_waveforms = []
            for w in waveforms:
                if self.draft.sample_rate != 48000:
                    w = torchaudio.functional.resample(
                        w.unsqueeze(0),
                        orig_freq=self.draft.sample_rate,
                        new_freq=48000
                    )[0]
                new_waveforms.append(w)
            waveforms = new_waveforms

            # 2) Verify
            if VERIFICATION_SWITCH == 1:
                verified_segments, first_reject = self._verify_waveforms_v1(waveforms, batch_chunks)
            elif VERIFICATION_SWITCH == 2:
                verified_segments, first_reject = self._verify_waveforms_v2(waveforms, batch_chunks)

            # 3) Keep accepted waveforms
            final_segments.extend(verified_segments)

            if first_reject == -1:
                # All accepted
                idx += self.speculation_batch_size
            else:
                # Re-generate that failing chunk with main
                global_fail_idx = idx + first_reject
                fail_text = chunks[global_fail_idx]
                print(f"[SpeculativeTTS] Fallback for chunk {global_fail_idx}: {fail_text[:50]}...")
                fallback_audio = self.main.generate(fail_text, seed=seed)
                final_segments.append(fallback_audio)
                idx = global_fail_idx + 1

        # 4) Crossfade
        return self._crossfade_segments(final_segments, self.sample_rate)
        # return torch.tensor([])

    def _draft_generate_batch(self, chunk_list, seed=42):
        """
        Generate waveforms for multiple chunks in parallel with the draft TTS.
        """
        waveforms = [None] * len(chunk_list)
        start_time = time.time()

        def worker(i, c):
            waveforms[i] = self.draft.generate(c, seed=seed)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = []
            for i, c in enumerate(chunk_list):
                futures.append(pool.submit(worker, i, c))
            for f in as_completed(futures):
                pass

        elapsed = time.time() - start_time
        print(f"[DraftTTS] Generated {len(chunk_list)} chunk(s) in {elapsed:.2f}s (batch).")

        return waveforms

    def _verify_waveforms_v1(self, waveforms, chunk_list):
        results = [None] * len(waveforms)
        
        for i in range(len(waveforms)):  # Process each waveform sequentially
            txt = chunk_list[i]
            w = waveforms[i]
            results[i] = self.verifier.verify_v1(txt, w, self.draft.sample_rate)
        # print(results)

        accepted_segments = []
        first_reject = -1

        for i, is_ok in enumerate(results):
            # if not is_ok and first_reject < 0:
            if not is_ok:
                first_reject = i
                break
            if is_ok:
                accepted_segments.append(waveforms[i])

        return accepted_segments, first_reject

    def _verify_waveforms_v2(self, waveforms, chunk_list):
        results = [None] * len(waveforms)
        
        for i in range(len(waveforms)):  # Process each waveform sequentially
            txt = chunk_list[i]
            w = waveforms[i]
            results[i] = self.verifier.verify_v2(txt, w, self.draft.sample_rate)
            print(results)

        accepted_segments = []
        first_reject = -1

        for i, is_ok in enumerate(results):
            if not is_ok and first_reject < 0:
                first_reject = i
                break
            if is_ok:
                accepted_segments.append(waveforms[i])

        return accepted_segments, first_reject


    def _split_text(self, text: str, max_chunk_len=MAX_CHUNK_LEN):
        raw_sentences = [s.strip() for s in text.split() if s.strip()]  # Split by spaces to get words
        chunks = []
        current = []

        for word in raw_sentences:
            if len(current) + 1 > max_chunk_len:  # If adding this word exceeds max_chunk_len
                chunks.append(" ".join(current))  # Store the current chunk
                current = [word]  # Start a new chunk
            else:
                current.append(word)

        if current:  # Add the last chunk if it's not empty
            chunks.append(" ".join(current))

        print(chunks, len(chunks))
        return chunks


    def _crossfade_segments(self, segments, sr: int) -> torch.Tensor:
        if not segments:
            return torch.tensor([], dtype=torch.float32)

        crossfade_len = int(sr * self.crossfade_sec)
        final_audio = segments[0]

        for seg in segments[1:]:
            if len(final_audio) < crossfade_len or len(seg) < crossfade_len:
                final_audio = torch.cat([final_audio, seg])  # If too short, just append
            else:
                fade_out = torch.linspace(1, 0, crossfade_len)
                fade_in = 1 - fade_out

                overlap_a = final_audio[-crossfade_len:]  # Last part of previous chunk
                overlap_b = seg[:crossfade_len]  # First part of new chunk
                
                xfade = (overlap_a * fade_out) + (overlap_b * fade_in)

                final_audio = torch.cat([
                    final_audio[:-crossfade_len],  # Remove overlapped part
                    xfade,  # Blended crossfade region
                    seg[crossfade_len:]  # Remaining part of new chunk
                ])

        return final_audio

#############################################
# 5) Demo / Comparison
#############################################
def run_comparison_demo():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42

    # -----------------------
    # Text to compare
    # -----------------------
    text = (
        "Hello everyone. This is a demonstration of speculative decoding for text to speech. "
        "We generate multiple chunks in parallel using a small TTS model, then use an ASR system to verify. "
        "If a chunk is correct, we accept it. Otherwise, we fall back to a larger TTS model. "
        "Finally, we merge the chunks with crossfading into a single waveform. "
        "We also produce two baselines for comparison. "
    )

    # -----------------------
    # 1) Load Models
    # -----------------------
    print("=== Loading Models ===")
    draft_model = DraftTTS(model_name="facebook/mms-tts-eng", device=device)
    main_model = MainTTS(model_name="OuteAI/OuteTTS-0.3-1B")
    verifier = ASRVerifier(model_name="openai/whisper-base", device=device)

    # -----------------------
    # 2) Generate - Draft Only
    # -----------------------
    print("\n=== 1) Draft-Only Baseline ===")
    t0 = time.time()
    draft_audio = draft_model.generate(text, seed=seed)
    t1 = time.time()
    draft_time = t1 - t0
    draft_dur = len(draft_audio) / draft_model.sample_rate
    print(f"[Draft-Only] Duration: {draft_dur:.2f}s | Time: {draft_time:.2f}s | RTF={draft_time/draft_dur:.2f}x")

    # -----------------------
    # 3) Generate - Main Only
    # -----------------------
    print("\n=== 2) Main-Only Baseline ===")
    t0 = time.time()
    main_audio = main_model.generate(text, seed=seed)
    t1 = time.time()
    main_time = t1 - t0
    main_dur = len(main_audio) / main_model.sample_rate

    print(f"[Main-Only] Duration: {main_dur:.2f}s | Time: {main_time:.2f}s | RTF={main_time/main_dur:.2f}x")

    # -----------------------
    # 4) Generate - Speculative
    # -----------------------
    print("\n=== 3) Speculative TTS ===")
    pipeline = SpeculativeTTSPipeline(
        draft_model=draft_model,
        main_model=main_model,
        verifier=verifier,
        speculation_batch_size=SPECULATION_BATCH_SIZE,
        crossfade_sec=0.05,
        device=device
    )

    t0 = time.time()
    spec_audio = pipeline.generate_speech(text, seed=seed)
    t1 = time.time()
    spec_time = t1 - t0
    spec_dur = len(spec_audio) / pipeline.sample_rate
    print(f"[Speculative] Duration: {spec_dur:.2f}s | Time: {spec_time:.2f}s | RTF={spec_time/spec_dur:.2f}x")

    # -----------------------
    # Save audio
    # -----------------------
    out_dir = "tts_output"
    os.makedirs(out_dir, exist_ok=True)

    # Draft only
    draft_path = os.path.join(out_dir, "draft_only.wav")
    torchaudio.save(draft_path, draft_audio.unsqueeze(0), draft_model.sample_rate)

    # Main only
    main_path = os.path.join(out_dir, "main_only.wav")
    torchaudio.save(main_path, main_audio.unsqueeze(0), main_model.sample_rate)

    # Speculative
    spec_path = os.path.join(out_dir, "speculative_output.wav")
    torchaudio.save(spec_path, spec_audio.unsqueeze(0), pipeline.sample_rate)

    print("\n=== Summary ===")
    print(f"Draft-Only: {draft_dur:.2f}s audio, took {draft_time:.2f}s (RTF={draft_time/draft_dur:.2f}x) -> {draft_path}")
    print(f"Main-Only : {main_dur:.2f}s audio, took {main_time:.2f}s (RTF={main_time/main_dur:.2f}x) -> {main_path}")
    print(f"Speculative: {spec_dur:.2f}s audio, took {spec_time:.2f}s (RTF={spec_time/spec_dur:.2f}x) -> {spec_path}")


if __name__ == "__main__":
    run_comparison_demo()
