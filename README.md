# TM Retrieval with DMD (Binary Amplitude Modulation)

Reconstruction of the Transmission Matrix (TM) of a scattering medium using binary amplitude modulation via a Digital Micromirror Device (DMD).

---

## Overview

This project measures the complex Transmission Matrix of a scattering medium using a DMD, which provides binary amplitude-only control (mirrors set to ON = 1 or OFF = 0). The TM is retrieved using the Gerchberg-Saxton (GS) algorithm adapted for amplitude-only modulation. A key challenge specific to DMD-based retrieval is correctly handling the DC term — the unmodulated background light that contributes to every measurement.

The retrieved TM is then validated by predicting output intensities for unseen test patterns and measuring Pearson correlation against experiment.

---

## Physical Setup

```
Laser → DMD → Lens system → Scattering medium → Camera
```

- **Light source:** Laser (collimated)
- **Modulator:** Digital Micromirror Device (DMD) — binary amplitude only
- **Lens system:** Focuses and images the DMD-modulated beam onto the scattering medium input face
- **Scattering medium:** Diffuse/turbid sample (e.g., ZnO layer, ground glass)
- **Detector:** CMOS/CCD camera capturing output speckle intensity patterns

---

## Workflow

### 1. Data Acquisition

- Divide the DMD active area into macropixels (groups of physical mirrors treated as one unit).
- Generate a set of `M` random binary amplitude patterns (each macropixel independently set to 0 or 1).
- Display each pattern on the DMD and record the resulting output intensity speckle pattern on the camera.
- Save all input patterns and corresponding camera frames.

### 2. TM Computation

- Apply the Gerchberg-Saxton algorithm adapted for **amplitude-only** modulation.
- Account for the DC term: since setting a macropixel to 0 still allows unmodulated light to leak, the effective input field is not simply the binary mask — the background contribution must be subtracted or modeled.
- Iterate until the retrieved TM converges (monitor residual between predicted and measured intensities).

### 3. Validation

- Reserve a set of `K` test patterns (not used during TM retrieval).
- Use the retrieved TM to predict output intensities for each test pattern.
- Compute the **Pearson correlation coefficient** between predicted and measured intensities for each test pattern.
- Report the mean correlation across test patterns as the accuracy metric.

---

## Parametric Study: Effect of Oversampling Ratio

The main variable is the number of input patterns `M` used for retrieval, expressed as a multiple of `N` (the number of controlled macropixels):

| Trial | Number of patterns |
|-------|--------------------|
| 1     | M = N              |
| 2     | M = 2N             |
| 3     | M = 4N             |

For each trial, retrieve the TM and evaluate prediction accuracy on the same held-out test set. Plot Pearson correlation vs. oversampling ratio `M/N` to characterize how accuracy improves with more measurements.

**Expected behavior:** Accuracy increases with oversampling ratio and saturates as the system becomes overdetermined.

---

## Bonus Experiment: PSF Engineering

Use the retrieved TM to engineer the output spatial mode:

- Choose a target output field (e.g., a donut-shaped intensity profile, a Laguerre-Gaussian mode, or any desired spatial pattern).
- Invert the TM to find the required input field that would produce the target output.
- Binarize the computed input field to obtain a valid DMD pattern (0/1 only).
- Display the binary pattern on the DMD and record the resulting output intensity.
- Compare the measured output to the target shape and quantify the match (e.g., mode overlap integral).

---

## Repository Structure

```
.
├── dmd_camera_control.ipynb    # controlling camera and DMD, generating dataset 
├── train.py                    # training on dataset using Gerchberg-Saxton algorithm and to retrieve the matrix
├── test.py                     # testing the transmission matrix and getting correlation
├── results/
│   └── tm_retrieved.npy      # Saved TM arrays and additional diagrams
└── README.md
```

---

## Dependencies

```
numpy
scipy
matplotlib
opencv-python
```

DMD and camera SDK libraries will depend on your specific hardware (e.g., ALP for Vialux DMDs, `instrumental-lib`, or vendor-provided Python APIs).

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `N` | Number of controlled macropixels (input DOF) |
| `macropixel_size` | Physical DMD mirrors per macropixel |
| `M` | Number of random binary patterns used for retrieval |
| `K` | Number of held-out test patterns for validation |
| `max_iter` | Maximum GS iterations |
| `tol` | Convergence threshold (residual change per iteration) |

---

## Results Summary

After running the parametric study, expected outputs include:

- Retrieved TM magnitude and phase maps for each oversampling ratio
- Pearson correlation coefficient vs. `M/N` curve
- (Bonus) Measured vs. target output intensity for PSF-engineered pattern

---

## References

- Popoff, S. M. et al. *Measuring the Transmission Matrix in Optics.* PRL 104, 100601 (2010)
- Conkey, D. B. et al. *Genetic algorithm optimization for focusing through turbid media.* Optics Express (2012)
- Gerchberg, R. W. & Saxton, W. O. *A practical algorithm for the determination of phase from image and diffraction plane pictures.* Optik (1972)
