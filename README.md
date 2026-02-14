# JON (Joseph's odd loader's)

simple set of nodes to bealbe to make my use case of comfyui a bit more simple. 

[ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) is required to use these nodes. 

## Key Features

* **JonLoader**: A unified "Super-Loader" that handles Checkpoints, GGUF, and Diffusion models in a single node. Includes built-in support for **SageAttention**, dual-VAE configurations (ideal for LTX2/Audio), and dynamic LoRA stacking.
* **JonWorkflowSettings**: The "Central Command" for your workspace. Manage global seeds, aspect ratios, resolutions (480p to 4K), and multi-image inputs (Start/Mid/End) from a single location.
* **Advanced Samplers**: Specialized samplers optimized for Flux, LTX2, Wan2.2, and Qwen, ensuring high-fidelity results with minimal node spaghetti.
* **GGUF Native**: Deep integration with GGUF model and CLIP loading for memory-efficient inference.

## Supported Models
Image models
* ZImage-Turbo
* Flux2 Klein 9b
* Qwen 2511

Video Models
* Wan2.2 Img2Vid
* LTX2


## Nodes

### 1. JonLoader

Loads all the models that are needed for the supported tyes of models

* **Model Support**: Seamlessly switch between `.safetensor` checkpoints, `.gguf` Unets, and standalone diffusion models.
* **SageAttention**: Directly apply SageAttention kernels (e.g., `sageattn3`, `qk_int8_pv_fp16`) for speedups on supported hardware.
* **Dynamic LoRA Stack**: Integrates with a JSON-based LoRA stack (compatible with `JobFloatSlider`) to manage multiple LoRAs without adding manual nodes for every file.
* **Dual VAE/Audio**: Supports a secondary VAE slot specifically for LTX2 audio VAEs or specialized precision modes (`fp32`, `bf16`).

### 2. JonWorkflowSettings

handles simple math for resolutions and aspect ratios while changing the input images to match

* **Resolution Logic**: Select a base resolution (e.g., `1080p`) and an aspect ratio (e.g., `21:9`), and the node automatically calculates the dimensions.
* **Triple-Image Input**: Load and process up to three images (Start, Mid, End) with built-in cropping rules: `PreserveAspectCrop`, `PreserveAspectFit`, `Stretch`, or `Center`.
* **Global Parameters**: Centralizes Positive/Negative prompts, Seed, and Video timing (Seconds to Frame count conversion).


### 3. JonChannelMixer
The `JonChannelMixer` provides a visual, tactile mixing console within the ComfyUI interface. It is designed to replace multiple individual float sliders with a single, space-efficient "fader bank," making it ideal for managing LoRA weights, IPAdapter strengths, or any multi-variable blending.

* **Visual Interface**: Features a custom-rendered UI with up to 8 fader strips, realistic fader caps, and dynamic color-coded grooves that change from blue to red based on the fader's percentage.
* **Tactile Controls**:
* **M (Mute)**: Instantly toggles the channel to 0.0 while remembering the previous value for quick A/B testing.
* **0 / 1 Buttons**: Dedicated snap-to buttons for instantly setting a channel to 0.0 or 1.0.
* **Value Overlays**: Displays precise real-time float values directly above each fader cap.
* **Dynamic Range**: Users can define global `min_val` and `max_val` limits (e.g., -1.0 to 1.0 or 0.0 to 10.0), and the faders automatically remap their travel distance to this range.
* **Ghost Inputs**: In the backend, the node converts standard widgets into "converted-widgets," hiding them from the standard node view to prevent clutter while keeping them accessible for automation or API calls.
* **API Compatibility**: While highly visual in the GUI, the node functions as a standard set of float outputs in API mode, ensuring that external calls can still manipulate individual channels via the `ch_1` through `ch_8` inputs.

## Installation

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/JosephMillsAtWork/JON.git
```

## Example Workflows

Check the `example_workflows` directory for JSON templates and preview images:

Image models
* ZImage-Turbo
* Flux2 Klein 9b
* Qwen 2511

Video Models
* Wan2.2 Img2Vid
* LTX2
