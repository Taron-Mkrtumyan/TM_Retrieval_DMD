"""
DMD control and camera capture.
"""

def display_pattern(pattern):
    """Display a binary pattern on the DMD."""
    raise NotImplementedError

def capture_frame():
    """Capture an intensity frame from the camera."""
    raise NotImplementedError

def acquire_dataset(patterns):
    """
    Display each pattern on the DMD and record the output intensity.
    Returns an array of camera frames.
    """
    frames = []
    for p in patterns:
        display_pattern(p)
        frames.append(capture_frame())
    return frames
