import torch
import time
import numpy as np
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXTransformer3DModel,
    CogVideoXPipeline,
)
from diffusers.utils import export_to_video
from transformers import T5EncoderModel
import os
import matplotlib.pyplot as plt
from datetime import timedelta

# Import quantization if available
try:
    from torchao.quantization import quantize_, int8_weight_only

    QUANTIZATION_AVAILABLE = True
except ImportError:
    print("TorchAO not available. Running without quantization.")
    QUANTIZATION_AVAILABLE = False


class SpeculativeVideoGenerator:
    def __init__(self, device="cuda", num_workers=1):
        """Initialize the speculative video generation system with main and draft models."""
        self.device = device
        self.num_workers = num_workers

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

        # Load the main model (5B)
        print("Loading main video model components (5B)...")
        text_encoder = T5EncoderModel.from_pretrained(
            "THUDM/CogVideoX1.5-5B",
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
        )
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            "THUDM/CogVideoX1.5-5B", subfolder="transformer", torch_dtype=torch.bfloat16
        )
        vae = AutoencoderKLCogVideoX.from_pretrained(
            "THUDM/CogVideoX1.5-5B", subfolder="vae", torch_dtype=torch.bfloat16
        )

        # Apply quantization if available
        if QUANTIZATION_AVAILABLE:
            print("Applying int8 weight-only quantization to main model...")
            quantize_(text_encoder, int8_weight_only())
            quantize_(transformer, int8_weight_only())
            quantize_(vae, int8_weight_only())

        # Create the main pipeline
        self.main_model = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX1.5-5B",
            text_encoder=text_encoder,
            transformer=transformer,
            vae=vae,
            torch_dtype=torch.bfloat16,
        )
        self.main_model.enable_model_cpu_offload()
        self.main_model.vae.enable_tiling()
        self.main_model.vae.enable_slicing()
        print(f"Main model (5B) loaded in {time.time() - load_start:.2f}s")

        # Load the 2B model for draft and verification
        print("Loading draft/verification model (2B)...")
        draft_load_start = time.time()
        draft_text_encoder = T5EncoderModel.from_pretrained(
            "THUDM/CogVideoX-2B",
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
        )
        draft_transformer = CogVideoXTransformer3DModel.from_pretrained(
            "THUDM/CogVideoX-2B", subfolder="transformer", torch_dtype=torch.bfloat16
        )
        draft_vae = AutoencoderKLCogVideoX.from_pretrained(
            "THUDM/CogVideoX-2B", subfolder="vae", torch_dtype=torch.bfloat16
        )

        # Apply quantization to draft model if available
        if QUANTIZATION_AVAILABLE:
            print("Applying int8 weight-only quantization to draft model...")
            quantize_(draft_text_encoder, int8_weight_only())
            quantize_(draft_transformer, int8_weight_only())
            quantize_(draft_vae, int8_weight_only())

        # Create the draft pipeline
        self.draft_model = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX-2B",
            text_encoder=draft_text_encoder,
            transformer=draft_transformer,
            vae=draft_vae,
            torch_dtype=torch.bfloat16,
        )
        self.draft_model.enable_model_cpu_offload()
        self.draft_model.vae.enable_tiling()
        self.draft_model.vae.enable_slicing()
        print(
            f"Draft/verification model (2B) loaded in {time.time() - draft_load_start:.2f}s"
        )
        print(f"Total model loading time: {time.time() - load_start:.2f}s")

    def generate_video(
        self, prompt, num_frames=16, output_dir="output_video", fps=8, create_video=True
    ):
        """Generate a video from a text prompt using speculative decoding."""
        overall_start_time = time.time()
        os.makedirs(output_dir, exist_ok=True)

        # Generate the first frame with the main model
        print(f"Generating initial frame with main model...")
        initial_frame = self._generate_initial_frame(prompt)
        final_frames = [initial_frame]

        # Process remaining frames with speculative decoding
        current_frame_idx = 1
        while current_frame_idx < num_frames:
            # Determine how many frames to speculate on
            frames_remaining = num_frames - current_frame_idx
            frames_to_speculate = min(self.num_workers * 2, frames_remaining)

            print(
                f"Speculating on frames {current_frame_idx} to {current_frame_idx + frames_to_speculate - 1}"
            )

            # Generate speculative frames with draft model
            speculative_frames = self._generate_speculative_frames(
                prompt, final_frames[-1], frames_to_speculate
            )

            # Verify frames with main model and accept until disagreement
            accepted_frames, num_accepted = self._verify_frames_with_main_model(
                prompt, final_frames[-1], speculative_frames
            )

            # Add accepted frames to final output
            final_frames.extend(accepted_frames)
            current_frame_idx += num_accepted

            # Update statistics
            self.timing_stats["accepted_chunks"] += num_accepted
            self.timing_stats["total_chunks"] += frames_to_speculate

            print(f"Accepted {num_accepted}/{frames_to_speculate} speculative frames")

            # If no frames were accepted, generate one frame with main model and continue
            if num_accepted == 0:
                print("No frames accepted, generating next frame with main model")
                next_frame = self._generate_next_frame_with_main_model(
                    prompt, final_frames[-1]
                )
                final_frames.append(next_frame)
                current_frame_idx += 1

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

    def _generate_initial_frame(self, prompt):
        """Generate the first frame with the main model."""
        start_time = time.time()

        with torch.no_grad():
            output = self.main_model(
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=15,
                num_frames=1,
                guidance_scale=2.0,
                height=32,
                width=32,
                generator=torch.Generator(device=self.device).manual_seed(42),
            ).frames[0][0]

        generation_time = time.time() - start_time
        self.timing_stats["main_generation"].append(generation_time)
        print(f"Initial frame generated in {generation_time:.2f}s")

        return output

    def _generate_speculative_frames(self, prompt, conditioning_frame, num_frames):
        """Generate speculative frames with the draft model."""
        start_time = time.time()

        # Check if the model requires frame counts to be multiples of 8
        if num_frames % 8 != 0:
            actual_num_frames = ((num_frames // 8) + 1) * 8
            print(
                f"Note: Adjusting requested frames from {num_frames} to {actual_num_frames} (multiple of 8)"
            )
        else:
            actual_num_frames = num_frames

        with torch.no_grad():
            output = self.draft_model(
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=1,  # Very few steps for draft model
                num_frames=actual_num_frames,
                guidance_scale=1.3,
                height=32,
                width=32,
                generator=torch.Generator(device=self.device).manual_seed(42),
            ).frames[0]

        # Only return the requested number of frames
        output = output[:num_frames]

        generation_time = time.time() - start_time
        self.timing_stats["draft_generation"].append(generation_time)
        print(
            f"Draft model generated {len(output)}/{actual_num_frames} frames in {generation_time:.2f}s"
        )

        return output

    def _verify_frames_with_main_model(
        self, prompt, conditioning_frame, speculative_frames
    ):
        """Verify speculative frames with the main model and accept until disagreement."""
        start_time = time.time()

        # Get the number of frames to verify
        num_frames = len(speculative_frames)

        # Check if the model requires frame counts to be multiples of 8
        if num_frames % 8 != 0:
            actual_num_frames = ((num_frames // 8) + 1) * 8
            print(
                f"  Note: Adjusting verification frames from {num_frames} to {actual_num_frames} (multiple of 8)"
            )
        else:
            actual_num_frames = num_frames

        # Generate reference frames with main model for comparison
        with torch.no_grad():
            reference_frames = self.main_model(
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=15,
                num_frames=actual_num_frames,
                guidance_scale=2.0,
                height=32,
                width=32,
                generator=torch.Generator(device=self.device).manual_seed(42),
            ).frames[0]

        # Only use the number of frames we need to compare
        reference_frames = reference_frames[:num_frames]

        # Debug print to check lengths
        print(
            f"  Debug: Speculative frames: {len(speculative_frames)}, Reference frames: {len(reference_frames)}"
        )

        # Compare each speculative frame with the corresponding reference frame
        accepted_frames = []
        for i in range(len(speculative_frames)):
            spec_frame = speculative_frames[i]
            ref_frame = reference_frames[i]

            # Convert frames to numpy arrays for comparison
            spec_np = np.array(spec_frame)
            ref_np = np.array(ref_frame)

            # Calculate probability-based acceptance using KL divergence or similar metric
            mse = np.mean((spec_np - ref_np) ** 2)
            normalized_mse = mse / (np.var(ref_np) + 1e-6)  # Normalize by variance

            # Convert to a "probability" (lower MSE = higher probability of acceptance)
            acceptance_prob = np.exp(
                -normalized_mse * 10
            )  # Scale factor for sensitivity

            print(
                f"  Frame {i}: MSE={mse:.4f}, Acceptance probability={acceptance_prob:.4f}"
            )

            # Accept if probability is above threshold
            if acceptance_prob > 0.7:  # Threshold for acceptance
                accepted_frames.append(spec_frame)
            else:
                # Stop at first disagreement (true speculative decoding)
                print(f"  Main model disagrees at frame {i}, stopping acceptance")
                break

        num_accepted = len(accepted_frames)
        verification_time = time.time() - start_time
        print(
            f"Verification completed in {verification_time:.2f}s, accepted {num_accepted} frames"
        )

        return accepted_frames, num_accepted

    def _generate_next_frame_with_main_model(self, prompt, conditioning_frame):
        """Generate the next frame with the main model when speculation fails."""
        start_time = time.time()

        with torch.no_grad():
            output = self.main_model(
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=15,
                num_frames=1,
                guidance_scale=2.0,
                height=32,
                width=32,
                generator=torch.Generator(device=self.device).manual_seed(42),
            ).frames[0][0]

        generation_time = time.time() - start_time
        self.timing_stats["main_generation"].append(generation_time)
        print(f"Next frame generated with main model in {generation_time:.2f}s")

        return output

    def _display_timing_statistics(self, total_time, num_frames):
        """Display detailed timing statistics."""
        # Calculate average times
        avg_draft_time = (
            np.mean(self.timing_stats["draft_generation"])
            if self.timing_stats["draft_generation"]
            else 0
        )
        avg_main_time = (
            np.mean(self.timing_stats["main_generation"])
            if self.timing_stats["main_generation"]
            else 0
        )
        avg_verify_time = (
            np.mean(self.timing_stats["verification"])
            if self.timing_stats["verification"]
            else 0
        )

        # Calculate acceptance rate
        acceptance_rate = (
            (self.timing_stats["accepted_chunks"] / self.timing_stats["total_chunks"])
            * 100
            if self.timing_stats["total_chunks"] > 0
            else 0
        )

        # Calculate theoretical time without speculation
        theoretical_sequential_time = avg_main_time * (
            self.timing_stats["total_chunks"] + 1
        )  # +1 for initial chunk

        # Calculate speedup
        speedup = theoretical_sequential_time / total_time if total_time > 0 else 0

        # Print statistics
        print("\n" + "=" * 50)
        print("TIMING STATISTICS")
        print("=" * 50)
        print(f"Total generation time: {total_time:.2f}s")
        print(f"Number of frames generated: {num_frames}")
        print(f"Average time per frame: {total_time/num_frames:.2f}s")
        print(f"Frames per second: {num_frames/total_time:.2f}")
        print("-" * 50)
        print(f"Draft model average generation time: {avg_draft_time:.2f}s")
        print(f"Main model average generation time: {avg_main_time:.2f}s")
        print(f"Average verification time: {avg_verify_time:.2f}s")
        print("-" * 50)
        print(
            f"Chunks accepted: {self.timing_stats['accepted_chunks']} / {self.timing_stats['total_chunks']} ({acceptance_rate:.1f}%)"
        )
        print(
            f"Estimated sequential generation time: {theoretical_sequential_time:.2f}s"
        )
        print(f"Speedup from speculation: {speedup:.2f}x")
        print("=" * 50)

        # Create a visualization of the timing data
        self._create_timing_visualization(total_time, theoretical_sequential_time)

    def _create_timing_visualization(self, actual_time, theoretical_time):
        """Create a visualization of timing data."""
        try:
            # Create a bar chart comparing actual vs theoretical time
            fig, ax = plt.subplots(figsize=(10, 6))

            # Data for the chart
            methods = ["Speculative Decoding", "Sequential Generation"]
            times = [actual_time, theoretical_time]

            # Create the bar chart
            bars = ax.bar(methods, times, color=["green", "red"])

            # Add labels and title
            ax.set_ylabel("Time (seconds)")
            ax.set_title("Speculative Decoding vs Sequential Generation")

            # Add time labels on top of bars
            for bar in bars:
                height = bar.get_height()
                time_str = str(timedelta(seconds=int(height)))
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.1,
                    f"{height:.1f}s\n({time_str})",
                    ha="center",
                    va="bottom",
                )

            # Add speedup text
            speedup = theoretical_time / actual_time if actual_time > 0 else 0
            ax.text(
                0.5,
                0.5,
                f"Speedup: {speedup:.2f}x",
                transform=ax.transAxes,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.8),
            )

            # Save the chart
            plt.tight_layout()
            plt.savefig("timing_comparison.png")
            print("Timing visualization saved to timing_comparison.png")

        except Exception as e:
            print(f"Error creating visualization: {e}")

    def generate_video_sequential(
        self, prompt, num_frames=16, output_dir="output_video", fps=8, create_video=True
    ):
        """Generate a video from a text prompt using sequential generation (no speculation)."""
        overall_start_time = time.time()
        os.makedirs(output_dir, exist_ok=True)

        # Generate the first frame with the main model
        print(f"Generating initial frame with main model...")
        initial_frame = self._generate_initial_frame(prompt)
        final_frames = [initial_frame]

        # Generate remaining frames sequentially with the main model
        for i in range(1, num_frames):
            print(f"Generating frame {i} with main model...")
            next_frame = self._generate_next_frame_with_main_model(
                prompt, final_frames[-1]
            )
            final_frames.append(next_frame)

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
        print("\n" + "=" * 50)
        print("SEQUENTIAL GENERATION STATISTICS")
        print("=" * 50)
        print(f"Total generation time: {total_time:.2f}s")
        print(f"Number of frames generated: {len(final_frames)}")
        print(f"Average time per frame: {total_time/len(final_frames):.2f}s")
        print(f"Frames per second: {len(final_frames)/total_time:.2f}")
        print("=" * 50)

        return final_frames


def generate_video_example():
    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create the generator with more workers for parallelism
    generator = SpeculativeVideoGenerator(device=device, num_workers=3)

    # Prompt to generate
    prompt = (
        "A little girl is riding a bicycle at high speed. Focused, detailed, realistic."
    )

    # Generate video with speculative decoding
    print("\n=== GENERATING VIDEO WITH SPECULATIVE DECODING ===\n")
    speculative_start = time.time()
    speculative_frames = generator.generate_video(
        prompt=prompt,
        num_frames=16,
        output_dir="output_video/speculative",
        fps=8,
        create_video=True,
    )
    speculative_time = time.time() - speculative_start

    # Generate video with sequential approach (no speculation)
    print("\n=== GENERATING VIDEO WITH SEQUENTIAL APPROACH (NO SPECULATION) ===\n")
    sequential_start = time.time()
    sequential_frames = generator.generate_video_sequential(
        prompt=prompt,
        num_frames=16,
        output_dir="output_video/sequential",
        fps=8,
        create_video=True,
    )
    sequential_time = time.time() - sequential_start

    # Print comparison
    print("\n" + "=" * 50)
    print("DIRECT COMPARISON")
    print("=" * 50)
    print(f"Sequential generation time: {sequential_time:.2f}s")
    print(f"Speculative generation time: {speculative_time:.2f}s")
    print(f"Speedup: {sequential_time/speculative_time:.2f}x")
    print("=" * 50)

    # Create side-by-side comparison video
    try:
        import subprocess

        print("\nCreating side-by-side comparison video...")

        # Use ffmpeg to create side-by-side video
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            "output_video/sequential/output_video.mp4",
            "-i",
            "output_video/speculative/output_video.mp4",
            "-filter_complex",
            "[0:v]pad=iw*2:ih[left];[left][1:v]overlay=w[vid];[vid]drawtext=text='Sequential: %.2fs':x=10:y=10:fontsize=24:fontcolor=white,drawtext=text='Speculative: %.2fs':x=w/2+10:y=10:fontsize=24:fontcolor=white"
            % (sequential_time, speculative_time),
            "output_video/comparison.mp4",
        ]

        subprocess.run(cmd, check=True)
        print("Side-by-side comparison video saved to output_video/comparison.mp4")
    except Exception as e:
        print(f"Error creating comparison video: {e}")

    return speculative_frames, sequential_frames


if __name__ == "__main__":
    generate_video_example()
