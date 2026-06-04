"""
DMD Data Acquisition Module
Project 2: TM Retrieval with Binary Amplitude Modulation

Handles:
- DMD pattern display (via ALP SDK or simulated fullscreen window)
- Camera frame capture
- Saving input/output pattern pairs
"""

import numpy as np
import cv2
import os
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    # DMD
    dmd_width: int = 1024          # DMD chip resolution (pixels)
    dmd_height: int = 768
    macropixel_size: int = 16      # group this many pixels per "input mode"
    dmd_exposure_us: int = 5000    # µs per frame (ALP SDK)

    # Camera
    camera_index: int = 0          # OpenCV camera index (or IP stream)
    cam_width: int = 1024
    cam_height: int = 1024
    cam_exposure_ms: float = 10.0
    cam_gain: float = 1.0
    n_avg_frames: int = 5          # frames averaged per pattern

    # Experiment
    output_dir: str = "data"
    n_patterns: int = 256          # number of random binary patterns to display
    n_test_patterns: int = 32      # held-out test set

    # Derived (computed in __post_init__)
    n_input_modes: int = 0         # N = (dmd_width//mp) * (dmd_height//mp)

    def __post_init__(self):
        nx = self.dmd_width  // self.macropixel_size
        ny = self.dmd_height // self.macropixel_size
        self.n_input_modes = nx * ny

    @property
    def grid_shape(self) -> Tuple[int, int]:
        return (self.dmd_height // self.macropixel_size,
                self.dmd_width  // self.macropixel_size)


# ─────────────────────────────────────────────────────────────────────────────
# DMD INTERFACE  (swap the stub for your real SDK)
# ─────────────────────────────────────────────────────────────────────────────

class DMDController:
    """
    Thin wrapper around the DMD hardware.

    Supported backends:
      'alp'       – Vialux ALP4 SDK  (import ALP4)
      'fullscreen'– display on a secondary monitor with OpenCV/pygame
      'sim'       – software simulation (unit tests / offline dev)
    """

    def __init__(self, cfg: ExperimentConfig, backend: str = "sim"):
        self.cfg = cfg
        self.backend = backend
        self._device = None
        self._window_name = "DMD_OUTPUT"

        if backend == "alp":
            self._init_alp()
        elif backend == "fullscreen":
            self._init_fullscreen()
        elif backend == "sim":
            print("[DMD] Simulation mode – no hardware needed.")
        else:
            raise ValueError(f"Unknown DMD backend: {backend}")

    # ── ALP SDK ──────────────────────────────────────────────────────────────
    def _init_alp(self):
        try:
            import ALP4
            self._device = ALP4.ALP4(version="4.3")
            self._device.Initialize()
            self._device.SetTiming(
                illuminationTime=self.cfg.dmd_exposure_us,
                pictureTime=self.cfg.dmd_exposure_us + 500
            )
            print(f"[DMD] ALP device initialised: {self._device.DevInquire(ALP4.ALP_DEV_DMDTYPE)}")
        except ImportError:
            raise RuntimeError("ALP4 SDK not found. Install it or use backend='fullscreen'.")

    def _alp_display(self, pattern: np.ndarray):
        """Send one binary pattern (H×W uint8, values 0/255) to ALP."""
        import ALP4
        flat = pattern.flatten().astype(np.uint8)
        seq_id = self._device.SeqAlloc(nbImg=1, bitDepth=1)
        self._device.SeqPut(imgData=flat, SeqId=seq_id)
        self._device.Run(SeqId=seq_id)
        time.sleep(self.cfg.dmd_exposure_us / 1e6 * 2)
        self._device.Halt()
        self._device.SeqFree(SeqId=seq_id)

    # ── Fullscreen window ────────────────────────────────────────────────────
    def _init_fullscreen(self):
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self._window_name,
                              cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
        # Move to second monitor – adjust (1920, 0) to your display offset
        cv2.moveWindow(self._window_name, 1920, 0)

    def _fullscreen_display(self, pattern: np.ndarray):
        bgr = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)
        cv2.imshow(self._window_name, bgr)
        cv2.waitKey(max(1, self.cfg.dmd_exposure_us // 1000))

    # ── Public API ───────────────────────────────────────────────────────────
    def display(self, pattern: np.ndarray):
        """
        Display pattern on DMD.
        pattern : (H, W) uint8 array, values 0 or 255
        """
        assert pattern.shape == (self.cfg.dmd_height, self.cfg.dmd_width)
        if self.backend == "alp":
            self._alp_display(pattern)
        elif self.backend == "fullscreen":
            self._fullscreen_display(pattern)
        # sim: do nothing

    def close(self):
        if self.backend == "alp" and self._device:
            self._device.Free()
        elif self.backend == "fullscreen":
            cv2.destroyWindow(self._window_name)


# ─────────────────────────────────────────────────────────────────────────────
# CAMERA INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class CameraController:
    """
    OpenCV-based camera wrapper.
    Replace grab() with your SDK calls for scientific cameras
    (e.g. Thorlabs, Basler, Hamamatsu).
    """

    def __init__(self, cfg: ExperimentConfig, simulate: bool = False):
        self.cfg = cfg
        self.simulate = simulate
        self._cap = None

        if not simulate:
            self._cap = cv2.VideoCapture(cfg.camera_index)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.cam_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_height)
            self._cap.set(cv2.CAP_PROP_EXPOSURE,    -cfg.cam_exposure_ms)
            self._cap.set(cv2.CAP_PROP_GAIN,         cfg.cam_gain)
            if not self._cap.isOpened():
                raise RuntimeError("Cannot open camera – check camera_index.")
        else:
            print("[Camera] Simulation mode.")

    def grab(self) -> np.ndarray:
        """Return a float32 intensity frame, shape (H, W), range [0, 1]."""
        if self.simulate:
            # Return structured noise for offline testing
            return np.random.rand(self.cfg.cam_height,
                                  self.cfg.cam_width).astype(np.float32)
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Camera grab failed.")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return gray / 255.0

    def grab_averaged(self) -> np.ndarray:
        """Average n_avg_frames to reduce shot noise."""
        frames = [self.grab() for _ in range(self.cfg.n_avg_frames)]
        return np.mean(frames, axis=0).astype(np.float32)

    def close(self):
        if self._cap:
            self._cap.release()


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def make_macropixel_pattern(binary_vec: np.ndarray,
                            cfg: ExperimentConfig) -> np.ndarray:
    """
    Expand a binary vector (N,) into a full DMD frame (H × W).
    Each macropixel covers cfg.macropixel_size × cfg.macropixel_size pixels.

    Returns uint8 array with values 0 or 255.
    """
    grid = binary_vec.reshape(cfg.grid_shape)          # (ny_mp, nx_mp)
    frame = np.kron(grid, np.ones((cfg.macropixel_size,
                                   cfg.macropixel_size), dtype=np.uint8))
    return (frame * 255).astype(np.uint8)


def generate_patterns(n: int, cfg: ExperimentConfig,
                      seed: int = 42) -> np.ndarray:
    """
    Generate n random binary patterns.
    Returns array of shape (n, N) where N = n_input_modes.
    Values are 0 or 1 (not scaled to 255 yet).
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=(n, cfg.n_input_modes),
                        dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# DATA ACQUISITION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class Acquisition:
    """
    Orchestrates: generate patterns → display on DMD → capture camera → save.
    """

    def __init__(self, cfg: ExperimentConfig,
                 dmd_backend: str = "sim",
                 cam_simulate: bool = True):
        self.cfg = cfg
        self.dmd = DMDController(cfg, backend=dmd_backend)
        self.cam = CameraController(cfg, simulate=cam_simulate)

        self.out_dir = Path(cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "inputs").mkdir(exist_ok=True)
        (self.out_dir / "outputs").mkdir(exist_ok=True)

        # Save config
        with open(self.out_dir / "config.json", "w") as f:
            json.dump(asdict(cfg), f, indent=2)

    # ── Single measurement ────────────────────────────────────────────────────
    def measure_one(self, binary_vec: np.ndarray,
                    idx: int, tag: str = "train") -> np.ndarray:
        """
        Display pattern, capture output, save both.
        Returns captured frame.
        """
        dmd_frame = make_macropixel_pattern(binary_vec, self.cfg)
        self.dmd.display(dmd_frame)

        # Brief settle time (adjust to your setup)
        time.sleep(0.002)

        output = self.cam.grab_averaged()

        # Save
        stem = f"{tag}_{idx:05d}"
        np.save(self.out_dir / "inputs"  / f"{stem}.npy", binary_vec)
        np.save(self.out_dir / "outputs" / f"{stem}.npy", output)

        # Optional PNG previews
        cv2.imwrite(str(self.out_dir / "inputs"  / f"{stem}.png"), dmd_frame)
        cv2.imwrite(str(self.out_dir / "outputs" / f"{stem}.png"),
                    (output * 255).astype(np.uint8))

        return output

    # ── Full acquisition run ──────────────────────────────────────────────────
    def run(self, seed_train: int = 42, seed_test: int = 99):
        """
        Acquire training + test pattern pairs and save index files.
        """
        n_total = self.cfg.n_patterns + self.cfg.n_test_patterns
        print(f"\n[Acquisition] Starting: {self.cfg.n_patterns} train "
              f"+ {self.cfg.n_test_patterns} test patterns")
        print(f"  DMD macropixels : {self.cfg.n_input_modes} "
              f"({self.cfg.grid_shape[1]}×{self.cfg.grid_shape[0]})")
        print(f"  Output dir      : {self.out_dir.resolve()}\n")

        # ── Training patterns ──
        train_vecs = generate_patterns(self.cfg.n_patterns, self.cfg, seed_train)
        train_outputs = []
        for i, vec in enumerate(train_vecs):
            out = self.measure_one(vec, i, tag="train")
            train_outputs.append(out)
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  Train [{i+1}/{self.cfg.n_patterns}]")

        # Save stacked arrays for fast loading
        np.save(self.out_dir / "train_inputs.npy",  train_vecs)
        np.save(self.out_dir / "train_outputs.npy", np.stack(train_outputs))

        # ── Test patterns ──
        test_vecs = generate_patterns(self.cfg.n_test_patterns, self.cfg, seed_test)
        test_outputs = []
        for i, vec in enumerate(test_vecs):
            out = self.measure_one(vec, i, tag="test")
            test_outputs.append(out)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Test  [{i+1}/{self.cfg.n_test_patterns}]")

        np.save(self.out_dir / "test_inputs.npy",  test_vecs)
        np.save(self.out_dir / "test_outputs.npy", np.stack(test_outputs))

        print("\n[Acquisition] Done. Data saved to:", self.out_dir.resolve())

    def close(self):
        self.dmd.close()
        self.cam.close()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = ExperimentConfig(
        dmd_width        = 1024,
        dmd_height       = 768,
        macropixel_size  = 32,    # 32×32 px macropixels → N = 32×24 = 768 modes
        n_patterns       = 768,   # 1× oversampling
        n_test_patterns  = 64,
        cam_width        = 512,
        cam_height       = 512,
        n_avg_frames     = 5,
        output_dir       = "data",
    )

    acq = Acquisition(cfg,
                      dmd_backend  = "sim",   # ← change to "alp" or "fullscreen"
                      cam_simulate = True)    # ← change to False for real camera
    try:
        acq.run()
    finally:
        acq.close()
