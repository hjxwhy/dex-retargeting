# EgoDex BrainCo Tools

## Install dependencies

This project uses `uv`. Install the example dependencies with exactly one PyTorch CUDA extra.

For CUDA 12.4 compatible machines:

```bash
uv sync --extra torch-cu124 --extra example
```

For CUDA 12.8 compatible machines:

```bash
uv sync --extra torch-cu128 --extra example
```

If `uv` cannot write to the default cache directory, use a writable cache path:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra torch-cu124 --extra example
```

The `example` extra installs viewer dependencies such as `viser[urdf]`, `rerun-sdk`, `opencv-python`, `mediapipe`, and `sapien`.

## Run viewers and converters

Use `uv run` so the scripts run inside the synced environment.

```bash
# Realtime visualization
uv run python example/egodex_brainco_tools/viser_brainco_hand_only_viewer.py \
  --hdf5 example/egodex_example/clean_cups/0.hdf5 \
  --fps 30 \
  --loop \
  --port 8080

# Convert data. Y-axis auto-centering is enabled by default.
uv run python example/egodex_brainco_tools/export_egodex_brainco_loop.py \
  --hdf5 example/egodex_example/clean_cups/1.hdf5 \
  --config example/egodex_brainco_tools/config/brainco_vector.yml \
  --output-dir example/egodex_example/clean_cups/0_brainco_loop \
  --loops 1

# Replay converted data
uv run python example/egodex_brainco_tools/viser_brainco_loop_json_viewer.py \
  --json example/egodex_example/clean_cups/0_brainco_loop/recomputed_ee_fullbody.json \
  --loop \
  --port 8080
```

If you still use a separate conda `egodex` environment, install the missing viewer dependency there manually:

```bash
python -m pip install "viser[urdf]"
```
