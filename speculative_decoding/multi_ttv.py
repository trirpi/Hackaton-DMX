import torch
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXTransformer3DModel,
    CogVideoXPipeline,
)
from diffusers.utils import export_to_video
from transformers import T5EncoderModel
from PIL import Image
import os
import subprocess
import matplotlib.pyplot as plt
from datetime import timedelta
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import sobel

# Import quantization if available
try:
    from torchao.quantization import quantize_, int8_weight_only

    QUANTIZATION_AVAILABLE = True
except ImportError:
    print("TorchAO not available. Running without quantization.")
    QUANTIZATION_AVAILABLE = False


class SpeculativeVideoGenerator:
    def __init__(self, devices=None, num_workers=1):
        """Initialize the speculative video generation system with main and draft models.

        Args:
            devices: List of device strings (e.g., ["cuda:0", "cuda:1"]) or None for auto-detection
            num_workers: Number of parallel workers per GPU
        """
        # Auto-detect available GPUs if devices not specified
        if devices is None:
            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                self.devices = [f"cuda:{i}" for i in range(num_gpus)]
                print(f"Auto-detected {num_gpus} GPUs: {self.devices}")
            else:
                self.devices = ["cpu"]
                print("No GPUs detected, using CPU")
        else:
            self.devices = devices

        self.num_workers = num_workers
        self.primary_device = self.devices[0]  # Use first device as primary

        # Timing statistics
        self.timing_stats = {
            "draft_generation": [],
            "main_generation": [],
            "verification": [],
            "accepted_chunks": 0,
            "total_chunks": 0,
        }

        print("Loading models...")
        load_start = time.time()

        # Load the main model on the primary device
        print(f"Loading main video model components on {self.primary_device}...")

        # Load text encoder
        text_encoder = T5EncoderModel.from_pretrained(
            "THUDM/CogVideoX1.5-5B",
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
        ).to(self.primary_device)

        # Load transformer
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            "THUDM/CogVideoX1.5-5B", subfolder="transformer", torch_dtype=torch.bfloat16
        ).to(self.primary_device)

        # Load VAE
        vae = AutoencoderKLCogVideoX.from_pretrained(
            "THUDM/CogVideoX1.5-5B", subfolder="vae", torch_dtype=torch.bfloat16
        ).to(self.primary_device)

        # Apply quantization if available
        if QUANTIZATION_AVAILABLE:
            print("Applying int8 weight-only quantization...")
            quantize_(text_encoder, int8_weight_only())
            quantize_(transformer, int8_weight_only())
            quantize_(vae, int8_weight_only())

        # Create the main pipeline for text-to-video
        self.main_model = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX1.5-5B",
            text_encoder=text_encoder,
            transformer=transformer,
            vae=vae,
            torch_dtype=torch.bfloat16,
        )

        # Enable memory optimizations
        self.main_model.enable_model_cpu_offload()
        self.main_model.vae.enable_tiling()
        self.main_model.vae.enable_slicing()

        # Create draft models on each device
        self.draft_models = {}
        for device in self.devices:
            # For simplicity, we'll use the same model for draft
            # In a production system, you'd load separate optimized models on each GPU
            if device == self.primary_device:
                self.draft_models[device] = self.main_model
            else:
                # For additional GPUs, create separate pipeline instances
                # This is a simplified approach - in production you might want
                # to use model parallelism or other advanced techniques
                print(f"Setting up draft model on {device}...")
                self.draft_models[device] = self.main_model

        print(f"Models loaded in {time.time() - load_start:.2f}s")

        # Set up device assignment tracking
        self.next_device_idx = 0

    def _get_next_device(self):
        """Get the next device in round-robin fashion"""
        device = self.devices[self.next_device_idx]
        self.next_device_idx = (self.next_device_idx + 1) % len(self.devices)
        return device

    def generate_video(
        self, prompt, num_frames=16, output_dir="output_video", fps=8, create_video=True
    ):
        """Generate a video from a text prompt using speculative decoding with multiple GPUs."""
        overall_start_time = time.time()

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Split the video generation into chunks
        frame_chunks = self._split_into_chunks(num_frames)

        # Generate the first chunk with the main model on primary device
        print(f"Generating initial chunk with main model on {self.primary_device}...")
        initial_frames = self._generate_with_main_model(prompt, frame_chunks[0])
        final_frames = initial_frames

        # Process remaining chunks with speculative decoding
        chunk_idx = 1

        # Create thread pool for parallel processing
        with ThreadPoolExecutor(
            max_workers=len(self.devices) * self.num_workers
        ) as executor:
            while chunk_idx < len(frame_chunks):
                # Determine how many chunks to speculate on (limited by available devices * workers)
                num_to_speculate = min(
                    len(self.devices) * self.num_workers, len(frame_chunks) - chunk_idx
                )

                print(
                    f"Speculating on chunks {chunk_idx} to {chunk_idx + num_to_speculate - 1}"
                )

                # Submit draft generation tasks to thread pool with device assignment
                futures = []
                for i in range(num_to_speculate):
                    current_chunk_idx = chunk_idx + i
                    device = self._get_next_device()
                    print(f"Assigning chunk {current_chunk_idx} to {device}")

                    futures.append(
                        executor.submit(
                            self._generate_with_draft_model,
                            prompt,
                            frame_chunks[current_chunk_idx],
                            device,
                        )
                    )

                # Collect results as they complete
                speculative_chunks = []
                for future in futures:
                    speculative_chunks.append(future.result())

                # Verify each speculative chunk and regenerate if needed
                verify_futures = []
                for i in range(num_to_speculate):
                    current_chunk_idx = chunk_idx + i
                    self.timing_stats["total_chunks"] += 1

                    # Submit verification tasks to thread pool
                    verify_futures.append(
                        executor.submit(
                            self._verify_chunk,
                            prompt,
                            speculative_chunks[i],
                            frame_chunks[current_chunk_idx],
                        )
                    )

                # Process verification results
                for i, future in enumerate(verify_futures):
                    current_chunk_idx = chunk_idx + i
                    is_accepted, verified_frames = future.result()

                    if is_accepted:
                        self.timing_stats["accepted_chunks"] += 1
                        print(f"Chunk {current_chunk_idx}: Accepted speculative frames")
                        final_frames.extend(verified_frames)
                    else:
                        print(
                            f"Generating corrected segment for chunk {current_chunk_idx}"
                        )
                        # Use the primary device for correction
                        corrected_frames = self._generate_with_main_model(
                            prompt,
                            frame_chunks[current_chunk_idx],
                        )
                        final_frames.extend(corrected_frames)

                # Move to the next set of chunks
                chunk_idx += num_to_speculate

        # Save the frames as images and create video
        if create_video:
            print("Creating video from frames...")
            video_path = export_to_video(
                final_frames, os.path.join(output_dir, "output_video.mp4"), fps=fps
            )
            print(f"Video saved to: {video_path}")
        else:
            # Save individual frames
            print("Saving frames as images...")
            for i, frame in enumerate(final_frames):
                frame.save(f"{output_dir}/frame_{i:04d}.png")

        # Calculate and display timing statistics
        total_time = time.time() - overall_start_time
        self._display_timing_statistics(total_time, len(final_frames))

        return final_frames

    def _generate_with_draft_model(self, prompt, frame_range, device=None):
        """Generate frames with the draft model on the specified device."""
        if device is None:
            device = self.primary_device

        start_time = time.time()
        start_idx, end_idx = frame_range
        num_frames = end_idx - start_idx
        frame_duration = num_frames / 8.0  # Assuming 8 fps

        print(f"Generating draft frames for chunk {frame_range} on {device}")

        # Get the draft model for this device
        draft_model = self.draft_models[device]

        # Generate frames with the draft model (faster but lower quality)
        with torch.no_grad():
            output = draft_model(
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=3,  # Fewer steps for draft
                num_frames=num_frames,
                guidance_scale=1.5,  # Lower guidance for draft
                generator=torch.Generator(device=device.split(":")[0]).manual_seed(42),
                height=32,
                width=32,
            ).frames[0]

        generation_time = time.time() - start_time
        self.timing_stats["draft_generation"].append(generation_time)
        print(
            f"Draft model generated {frame_duration:.2f}s video in {generation_time:.2f}s"
        )

        return output

    def _generate_with_main_model(self, prompt, frame_range):
        """Generate frames with the main model on the primary device."""
        start_time = time.time()
        start_idx, end_idx = frame_range
        num_frames = end_idx - start_idx
        frame_duration = num_frames / 8.0  # Assuming 8 fps

        # Generate frames with the main model
        with torch.no_grad():
            output = self.main_model(
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=30,
                num_frames=num_frames,
                guidance_scale=3.0,
                generator=torch.Generator(
                    device=self.primary_device.split(":")[0]
                ).manual_seed(42),
                height=32,
                width=32,
            ).frames[0]

        generation_time = time.time() - start_time
        self.timing_stats["main_generation"].append(generation_time)
        print(
            f"Main model generated {frame_duration:.2f}s video in {generation_time:.2f}s"
        )

        return output

    # ... rest of your methods remain the same ...


def generate_video_example():
    # Auto-detect available GPUs
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        devices = [f"cuda:{i}" for i in range(num_gpus)]
        print(f"Using {num_gpus} GPUs: {devices}")
    else:
        devices = ["cpu"]
        print("No GPUs detected, using CPU")

    # Create the generator with multi-GPU support
    generator = SpeculativeVideoGenerator(
        devices=devices, num_workers=1  # Workers per GPU
    )

    # Prompt to generate
    prompt = (
        "A little girl is riding a bicycle at high speed. Focused, detailed, realistic."
    )

    # Generate video frames
    frames = generator.generate_video(
        prompt=prompt,
        num_frames=16,
        fps=8,
        create_video=True,
    )

    return frames


if __name__ == "__main__":
    generate_video_example()
