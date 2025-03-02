import torch
from diffusers import (
    CogVideoXPipeline,
    AutoencoderKLCogVideoX,
    CogVideoXTransformer3DModel,
)
from transformers import T5EncoderModel
from torchao.quantization import quantize_, int8_weight_only
import time
import numpy as np
from torchvision import transforms
import subprocess
import os
import sys
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


def regular_generation(
    prompt, height=512, width=512, num_frames=16, steps=30, guidance=6.0, seed=42
):
    print("\nRunning regular generation...")

    # Load and quantize models
    text_encoder = T5EncoderModel.from_pretrained(
        "THUDM/CogVideoX-5b", subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    quantize_(text_encoder, int8_weight_only())

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        "THUDM/CogVideoX-5b", subfolder="transformer", torch_dtype=torch.bfloat16
    )
    quantize_(transformer, int8_weight_only())

    vae = AutoencoderKLCogVideoX.from_pretrained(
        "THUDM/CogVideoX-5b", subfolder="vae", torch_dtype=torch.bfloat16
    )
    quantize_(vae, int8_weight_only())

    # Create pipeline with quantized components
    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-5b",
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        torch_dtype=torch.bfloat16,
    )

    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    start_time = time.time()

    frames = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_videos_per_prompt=1,
        num_inference_steps=steps,
        num_frames=num_frames,
        guidance_scale=guidance,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).frames[0]

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Regular generation time: {total_time:.2f} seconds")

    return frames, total_time


def speculative_generation(
    prompt,
    height=512,
    width=512,
    num_frames=16,
    main_steps=2,
    draft_steps=2,
    main_guidance=6.0,
    draft_guidance=3.0,
    refinement_threshold=0.7,
    seed=42,
):
    print("\nRunning progressive refinement speculative generation...")

    # Load and quantize main model components
    main_text_encoder = T5EncoderModel.from_pretrained(
        "THUDM/CogVideoX-5b", subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    quantize_(main_text_encoder, int8_weight_only())

    main_transformer = CogVideoXTransformer3DModel.from_pretrained(
        "THUDM/CogVideoX-5b", subfolder="transformer", torch_dtype=torch.bfloat16
    )
    quantize_(main_transformer, int8_weight_only())

    main_vae = AutoencoderKLCogVideoX.from_pretrained(
        "THUDM/CogVideoX-5b", subfolder="vae", torch_dtype=torch.bfloat16
    )
    quantize_(main_vae, int8_weight_only())

    # Create main pipeline with quantized components
    main_pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-5b",
        text_encoder=main_text_encoder,
        transformer=main_transformer,
        vae=main_vae,
        torch_dtype=torch.bfloat16,
    )

    # Load and quantize draft model components
    draft_text_encoder = T5EncoderModel.from_pretrained(
        "THUDM/CogVideoX-2b", subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    quantize_(draft_text_encoder, int8_weight_only())

    draft_transformer = CogVideoXTransformer3DModel.from_pretrained(
        "THUDM/CogVideoX-2b", subfolder="transformer", torch_dtype=torch.bfloat16
    )
    quantize_(draft_transformer, int8_weight_only())

    draft_vae = AutoencoderKLCogVideoX.from_pretrained(
        "THUDM/CogVideoX-2b", subfolder="vae", torch_dtype=torch.bfloat16
    )
    quantize_(draft_vae, int8_weight_only())

    # Create draft pipeline with quantized components
    draft_pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-2b",
        text_encoder=draft_text_encoder,
        transformer=draft_transformer,
        vae=draft_vae,
        torch_dtype=torch.bfloat16,
    )

    # Enable optimizations
    for pipe in [main_pipe, draft_pipe]:
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()

    start_time = time.time()

    # 1. Generate all frames with draft model first
    print(f"Generating all frames with draft model ({draft_steps} steps)...")
    draft_start = time.time()
    draft_output = draft_pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_videos_per_prompt=1,
        num_inference_steps=draft_steps,
        num_frames=num_frames,
        guidance_scale=draft_guidance,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    )
    draft_end = time.time()
    draft_time = draft_end - draft_start
    print(f"Draft model generation time: {draft_time:.2f} seconds")

    draft_frames = draft_output.frames[0]

    # 2. Identify frames that need refinement using multiple metrics
    print("Analyzing frames for refinement using advanced metrics...")
    frames_to_refine = []
    frame_scores = []

    # Convert frames to tensors for analysis
    to_tensor = transforms.ToTensor()
    draft_tensors = [to_tensor(frame) for frame in draft_frames]

    # A. Temporal consistency metric (frame-to-frame differences)
    temporal_scores = []
    for i in range(1, len(draft_tensors)):
        # Mean squared error between consecutive frames
        mse = torch.mean((draft_tensors[i] - draft_tensors[i - 1]) ** 2).item()

        # Structural similarity (approximation)
        prev_std = torch.std(draft_tensors[i - 1])
        curr_std = torch.std(draft_tensors[i])
        prev_mean = torch.mean(draft_tensors[i - 1])
        curr_mean = torch.mean(draft_tensors[i])
        covariance = torch.mean(
            (draft_tensors[i] - curr_mean) * (draft_tensors[i - 1] - prev_mean)
        )
        ssim_approx = covariance / (prev_std * curr_std + 1e-8)
        ssim_score = (
            1 - ssim_approx.item()
        ) / 2  # Convert to 0-1 range where lower is better

        # Combined temporal score (weighted average)
        temporal_score = 0.7 * mse + 0.3 * ssim_score
        temporal_scores.append(temporal_score)

    # B. Content quality metrics
    content_scores = []
    for tensor in draft_tensors:
        # Simplified sharpness metric using Laplacian approximation
        # Center - surrounding pixels (simplified Laplacian)
        laplacian = torch.zeros_like(tensor)
        laplacian[:, 1:-1, 1:-1] = 4 * tensor[:, 1:-1, 1:-1] - (
            tensor[:, :-2, 1:-1]
            + tensor[:, 2:, 1:-1]
            + tensor[:, 1:-1, :-2]
            + tensor[:, 1:-1, 2:]
        )
        sharpness = torch.mean(torch.abs(laplacian)).item()
        sharpness_score = 1.0 - min(sharpness, 1.0)  # Normalize and invert

        # Contrast metric
        contrast = torch.std(tensor).item()
        contrast_score = 1.0 - contrast  # Lower contrast -> higher score

        # Combined content score
        content_score = 0.6 * sharpness_score + 0.4 * contrast_score
        content_scores.append(content_score)

    # C. Combine metrics and identify frames to refine
    for i in range(len(draft_frames)):
        # For first frame, only use content score
        if i == 0:
            final_score = content_scores[i]
        # For last frame, use content score with higher weight
        elif i == len(draft_frames) - 1:
            final_score = 0.7 * content_scores[i] + 0.3 * temporal_scores[i - 1]
        # For middle frames, use both temporal and content scores
        else:
            final_score = 0.5 * content_scores[i] + 0.5 * temporal_scores[i - 1]

        frame_scores.append(final_score)

    # Calculate threshold based on scores distribution
    mean_score = np.mean(frame_scores)
    std_score = np.std(frame_scores)
    threshold = mean_score + refinement_threshold * std_score

    # Identify frames to refine
    for i in range(len(draft_frames)):
        if frame_scores[i] > threshold:
            frames_to_refine.append(i)
            # Also refine adjacent frames if they're not already included
            if i > 0 and i - 1 not in frames_to_refine:
                frames_to_refine.append(i - 1)
            if i < len(draft_frames) - 1 and i + 1 not in frames_to_refine:
                frames_to_refine.append(i + 1)

    # Always include first and last frames for refinement
    if 0 not in frames_to_refine:
        frames_to_refine.append(0)
    if len(draft_frames) - 1 not in frames_to_refine:
        frames_to_refine.append(len(draft_frames) - 1)

    # Remove duplicates and sort
    frames_to_refine = sorted(list(set(frames_to_refine)))

    # Print detailed metrics for presentation
    print("\nFrame-by-frame quality analysis:")
    for i in range(len(draft_frames)):
        status = "REFINE" if i in frames_to_refine else "KEEP"
        if i == 0:
            print(f"Frame {i:2d}: Score {frame_scores[i]:.4f} [{status}] (First frame)")
        elif i == len(draft_frames) - 1:
            print(f"Frame {i:2d}: Score {frame_scores[i]:.4f} [{status}] (Last frame)")
        else:
            temporal = temporal_scores[i - 1]
            content = content_scores[i]
            print(
                f"Frame {i:2d}: Score {frame_scores[i]:.4f} [{status}] (Temporal: {temporal:.4f}, Content: {content:.4f})"
            )

    print(
        f"\nIdentified {len(frames_to_refine)} frames for refinement: {frames_to_refine}"
    )

    # 3. Refine selected frames with main model
    refined_frames = draft_frames.copy()
    main_frame_times = []

    for idx in frames_to_refine:
        print(f"Refining frame {idx} with main model ({main_steps} steps)...")
        main_start = time.time()
        main_output = main_pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_videos_per_prompt=1,
            num_inference_steps=main_steps,
            num_frames=1,
            guidance_scale=main_guidance,
            generator=torch.Generator(device="cuda").manual_seed(seed + idx),
        )
        main_end = time.time()
        main_frame_time = main_end - main_start
        main_frame_times.append(main_frame_time)

        refined_frames[idx] = main_output.frames[0][0]

    end_time = time.time()
    total_time = end_time - start_time

    # Calculate statistics
    draft_frames_kept = num_frames - len(frames_to_refine)
    main_frames_used = len(frames_to_refine)

    print(f"\nFinal Statistics:")
    print(f"Total generation time: {total_time:.2f} seconds")
    print(f"Draft frames kept: {draft_frames_kept}")
    print(f"Main model refinements: {main_frames_used}")
    print(f"Refinement rate: {(main_frames_used/num_frames)*100:.1f}%")

    # Calculate theoretical time savings based on actual measurements
    draft_time_per_frame = draft_time / num_frames
    main_time_per_frame = (
        sum(main_frame_times) / len(main_frame_times) if main_frame_times else 0
    )

    theoretical_time = (num_frames * draft_time_per_frame) + (
        main_frames_used * main_time_per_frame
    )
    naive_time = num_frames * main_time_per_frame

    print(f"Theoretical time without overhead: {theoretical_time:.2f} seconds")
    print(f"Naive generation time (all main model): {naive_time:.2f} seconds")
    print(f"Theoretical speedup: {naive_time/theoretical_time:.2f}x")

    return refined_frames, total_time


def display_video(filename):
    """Open the video file with the default video player"""
    if os.name == "nt":  # for Windows
        os.startfile(filename)
    else:  # for Linux/Mac
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])


if __name__ == "__main__":
    # Test prompt
    prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."

    # Generate videos with both methods
    # First run regular generation
    reg_frames, reg_time = regular_generation(
        prompt=prompt,
        height=512,
        width=512,
        num_frames=16,
        steps=30,
        guidance=6.0,
        seed=42,
    )

    # Then run speculative generation with reg_time
    spec_frames, spec_time = speculative_generation(
        prompt=prompt,
        height=512,
        width=512,
        num_frames=16,
        main_steps=2,
        draft_steps=2,
        main_guidance=4.0,
        draft_guidance=3.0,
        refinement_threshold=0.3,
        seed=42,
    )

    # Print comparison
    print("\nPerformance Comparison:")
    print(f"Regular generation time: {reg_time:.2f} seconds")
    print(f"Speculative generation time: {spec_time:.2f} seconds")
    print(f"Speedup: {reg_time/spec_time:.2f}x")

    # Export both videos
    from diffusers.utils import export_to_video

    export_to_video(spec_frames, "speculative_output.mp4", fps=8)
    export_to_video(reg_frames, "regular_output.mp4", fps=8)

    # Create a directory for visualizations
    os.makedirs("visualizations", exist_ok=True)

    # Performance comparison chart
    plt.figure(figsize=(10, 8))
    methods = ["Speculative Decoding", "Sequential Generation"]
    times = [spec_time, reg_time]
    speedup = reg_time / spec_time if spec_time > 0 else 0

    # Convert seconds to minutes:seconds format
    time_labels = [f"{t:.1f}s\n({int(t//60):02d}:{int(t%60):02d})" for t in times]

    # Use green for speculative (faster) and red for sequential
    colors = ["#2ca02c", "#d62728"]  # Green, Red

    bars = plt.bar(methods, times, color=colors)

    # Add time labels on top of bars
    for i, (bar, label) in enumerate(zip(bars, time_labels)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            label,
            ha="center",
            va="bottom",
        )

    # Add speedup annotation in the middle
    if speedup > 1:
        plt.annotate(
            f"Speedup: {speedup:.2f}x",
            xy=(0.5, min(times) + (max(times) - min(times)) / 2),
            xytext=(0.5, min(times) + (max(times) - min(times)) / 2),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    plt.ylabel("Time (seconds)")
    plt.title("Speculative Decoding vs Sequential Generation")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("visualizations/performance_comparison.png", dpi=300)
    plt.close()

    print(
        "Performance comparison visualization saved to 'visualizations/performance_comparison.png'"
    )
