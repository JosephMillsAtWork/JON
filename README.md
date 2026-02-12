# My Production Tools for ComfyUI

Custom nodes designed to orchestrate video workflows (Wan2.2, LTX, Flux).

## Nodes

### 🎬 Production Director (3-Stage)
A master controller for resolution and timing.
- **Resolution:** Automatically handles 16:9, 4:3, Portrait/Landscape.
- **Safety:** Forces Width/Height to be divisible by 32 (prevents tensor mismatch errors).
- **Timing:** Calculates frame counts based on `(Seconds * FPS) + 1`.
- **Images:** Handles resizing for Start/Middle/End images to match the generated geometry.
