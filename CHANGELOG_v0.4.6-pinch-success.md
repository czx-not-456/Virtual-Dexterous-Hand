# v0.4.6-pinch-success

This patch keeps the v0.4.6 hand geometry and control unchanged.

The evaluation semantics are corrected for precision PINCH:
- PINCH success: bilateral thumb/index fingertip contact sustained for the configured streak (default 12 frames).
- PINCH stability: still reported separately by `stable_ratio` using the existing orientation-drift criterion.
- WRAP success: unchanged; still requires consecutive stable-contact frames.

This avoids classifying an otherwise valid precision pinch as a complete failure solely because the free object tilts more than 3 degrees from its initial orientation.
