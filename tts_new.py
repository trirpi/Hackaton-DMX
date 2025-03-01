import torch
import torchaudio
import numpy as np
from transformers import AutoTokenizer, VitsModel
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch.nn.functional as F
import time


class SpeculativeTTS:
    def __init__(self, device="cuda"):
        """
        Initialize the speculative TTS system with Facebook's MMS-TTS model.
        We'll use the same model architecture for both main and draft models,
        but with different generation parameters.
        """
        self.device = device

        # Load the main TTS model (high quality)
        print("Loading main MMS-TTS model...")
        self.main_model = VitsModel.from_pretrained("facebook/mms-tts-eng").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

        # For a true implementation, we would use a smaller/faster model as the draft model
        # For this example, we'll use the same model but with different inference parameters
        print("Loading draft MMS-TTS model...")
        self.draft_model = VitsModel.from_pretrained("facebook/mms-tts-eng").to(device)

        # Audio parameters
        self.sample_rate = 16000  # MMS-TTS sample rate

        # Speculative decoding parameters
        self.speculation_length = 3  # Number of chunks to speculate ahead
        self.max_workers = 4  # Maximum number of parallel workers

    def synthesize(self, text):
        """
        Synthesize speech from text using speculative decoding.

        Args:
            text: Input text to synthesize

        Returns:
            Synthesized audio waveform
        """
        # Split text into chunks
        text_chunks = self._split_text(text)
        print(f"Split text into {len(text_chunks)} chunks")

        # Final audio segments
        final_audio_segments = []

        # Process chunks with speculative decoding
        chunk_index = 0
        while chunk_index < len(text_chunks):
            # Determine how many chunks to speculate on
            remaining_chunks = len(text_chunks) - chunk_index
            num_to_speculate = min(self.speculation_length, remaining_chunks)

            print(
                f"Speculating on chunks {chunk_index} to {chunk_index + num_to_speculate - 1}"
            )

            # Generate speculative audio segments in parallel using the draft model
            speculative_segments = self._generate_speculative_segments(
                text_chunks[chunk_index : chunk_index + num_to_speculate]
            )

            # Verify speculative segments using the main model
            verified_segments, first_rejection = self._verify_speculative_segments(
                speculative_segments,
                text_chunks[chunk_index : chunk_index + num_to_speculate],
            )

            # Add verified segments to final audio
            final_audio_segments.extend(verified_segments)

            # Update chunk index based on verification results
            if first_rejection == -1:
                # All chunks were accepted
                chunk_index += num_to_speculate
            else:
                # Some chunks were rejected, move to the first rejected chunk
                chunk_index += first_rejection

                # Generate a corrected segment for the rejected chunk using the main model
                print(f"Generating corrected segment for chunk {chunk_index}")
                corrected_segment = self._generate_with_main_model(
                    text_chunks[chunk_index]
                )
                final_audio_segments.append(corrected_segment)
                chunk_index += 1

        # Assemble final audio
        final_audio = self._assemble_audio(final_audio_segments)

        return final_audio

    def _split_text(self, text):
        """Split text into manageable chunks for parallel processing"""
        # Split by sentences, then ensure chunks aren't too long
        sentences = [s.strip() for s in text.split(".") if s.strip()]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence would make the chunk too long, start a new chunk
            if len(current_chunk) + len(sentence) > 100:  # Arbitrary limit of 100 chars
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _generate_speculative_segments(self, text_chunks):
        """
        Generate speculative audio segments for multiple text chunks in parallel using the draft model.
        This is the core of the speculative decoding approach.
        """
        speculative_segments = []

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit generation tasks for each chunk
            futures = []
            for chunk in text_chunks:
                futures.append(executor.submit(self._generate_with_draft_model, chunk))

            # Collect results as they complete
            for future in as_completed(futures):
                speculative_segments.append(future.result())

        return speculative_segments

    def _generate_with_draft_model(self, text_chunk):
        """Generate audio with the draft model (faster but lower quality)"""
        start_time = time.time()

        # Tokenize the text
        inputs = self.tokenizer(text_chunk, return_tensors="pt").to(self.device)

        # Generate audio with draft model
        with torch.no_grad():
            # Simply call the model with the inputs
            output = self.draft_model(**inputs)

        # Get the waveform from the output (keep on device)
        waveform = output.waveform[0]

        print(
            f"Draft model generated {len(waveform)/self.sample_rate:.2f}s audio in {time.time()-start_time:.2f}s"
        )

        return waveform

    def _generate_with_main_model(self, text_chunk):
        """Generate audio with the main model (slower but higher quality)"""
        start_time = time.time()

        # Tokenize the text
        inputs = self.tokenizer(text_chunk, return_tensors="pt").to(self.device)

        # Generate audio with main model
        with torch.no_grad():
            # Simply call the model with the inputs
            output = self.main_model(**inputs)

        # Get the waveform from the output (keep on device)
        waveform = output.waveform[0]

        print(
            f"Main model generated {len(waveform)/self.sample_rate:.2f}s audio in {time.time()-start_time:.2f}s"
        )

        return waveform

    def _verify_speculative_segments(self, speculative_segments, text_chunks):
        """
        Verify speculative segments using the main model.
        Returns verified segments and the index of the first rejection (-1 if all accepted).
        """
        verified_segments = []
        first_rejection = -1

        # Process each segment in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit verification tasks
            futures = []
            for i, (segment, chunk) in enumerate(
                zip(speculative_segments, text_chunks)
            ):
                futures.append(executor.submit(self._verify_segment, segment, chunk, i))

            # Collect and sort results
            results = []
            for future in as_completed(futures):
                results.append(future.result())

            # Sort by index
            results.sort(key=lambda x: x[0])

            # Process results in order
            for i, is_accepted, segment in results:
                if not is_accepted and first_rejection == -1:
                    first_rejection = i
                    break

                if is_accepted:
                    verified_segments.append(segment)

        return verified_segments, first_rejection

    def _verify_segment(self, draft_segment, text_chunk, index):
        """
        Verify a single audio segment by comparing it with what the main model would generate.
        Returns (index, is_accepted, segment).
        """
        start_time = time.time()

        # Generate what the main model would produce for this chunk
        inputs = self.tokenizer(text_chunk, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # Simply call the model with the inputs
            main_output = self.main_model(**inputs)

        main_segment = main_output.waveform[0]

        # Ensure both segments are the same length for comparison
        min_length = min(len(draft_segment), len(main_segment))
        draft_segment_trimmed = draft_segment[:min_length]
        main_segment_trimmed = main_segment[:min_length]

        # Calculate similarity between draft and main model outputs
        # We use cosine similarity on the waveforms
        draft_tensor = draft_segment_trimmed.float()
        main_tensor = main_segment_trimmed.float()

        # Normalize the waveforms
        draft_norm = draft_tensor / (torch.norm(draft_tensor) + 1e-8)
        main_norm = main_tensor / (torch.norm(main_tensor) + 1e-8)

        # Calculate cosine similarity
        similarity = torch.dot(draft_norm, main_norm).item()

        # Determine if segment is accepted based on similarity threshold
        is_accepted = similarity > 0.7  # Adjust threshold as needed

        print(
            f"Chunk {index}: Similarity {similarity:.2f} - {'Accepted' if is_accepted else 'Rejected'}"
        )
        print(f"  Verification took {time.time()-start_time:.2f}s")

        return index, is_accepted, draft_segment

    def _assemble_audio(self, audio_segments):
        """Assemble audio segments with smooth crossfading"""
        if not audio_segments:
            return torch.tensor([])

        if len(audio_segments) == 1:
            return audio_segments[0]

        # Crossfade length (in samples)
        crossfade_length = int(0.05 * self.sample_rate)  # 50ms crossfade

        # Initialize the final audio with the first segment
        final_audio = audio_segments[0]

        # Add each subsequent segment with crossfading
        for segment in audio_segments[1:]:
            # If either segment is too short for crossfading, just concatenate
            if len(final_audio) < crossfade_length or len(segment) < crossfade_length:
                final_audio = torch.cat([final_audio, segment])
                continue

            # Create crossfade weights and move to the same device as final_audio
            fade_out = torch.linspace(1, 0, crossfade_length).to(final_audio.device)
            fade_in = torch.linspace(0, 1, crossfade_length).to(final_audio.device)

            # Ensure segment is on the same device as final_audio
            segment = segment.to(final_audio.device)

            # Apply crossfade
            final_end = final_audio[-crossfade_length:]
            segment_start = segment[:crossfade_length]

            crossfade = (final_end * fade_out) + (segment_start * fade_in)

            # Combine segments with crossfade
            final_audio = torch.cat(
                [final_audio[:-crossfade_length], crossfade, segment[crossfade_length:]]
            )

        return final_audio


def generate_speech_example():
    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create the generator
    generator = SpeculativeTTS(device=device)

    # Text to synthesize
    text = """
    This is a test of speculative voice generation. It should produce high-quality speech efficiently.
    The system uses Facebook's MMS-TTS model for both draft generation and verification.
    Speculative decoding allows us to generate multiple chunks in parallel and verify them quickly.
    This approach can significantly speed up the generation process while maintaining quality.
    """

    # Measure performance
    start_time = time.time()

    # Generate speech
    audio = generator.synthesize(text)

    # Calculate total time
    total_time = time.time() - start_time

    print(
        f"Generated {len(audio)/generator.sample_rate:.2f}s of audio in {total_time:.2f}s"
    )
    print(f"Real-time factor: {total_time/(len(audio)/generator.sample_rate):.2f}x")

    # Move audio to CPU before saving
    audio_cpu = audio.cpu()

    # Save the audio
    torchaudio.save("output_speech.wav", audio_cpu.unsqueeze(0), generator.sample_rate)

    return audio


if __name__ == "__main__":
    audio = generate_speech_example()
