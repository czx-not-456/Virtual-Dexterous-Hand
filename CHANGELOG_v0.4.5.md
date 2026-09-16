# v0.4.5 — PINCH approach stabilization

This version keeps the verified WRAP controller and combines two PINCH fixes:

1. **Larger calibration block**: PINCH box half-size is `[0.045, 0.015, 0.066]`, i.e. a physical 90 × 30 × 132 mm block. The bottom remains on the z=0.070 m table.
2. **Approach fixture**: while PINCH is active and bilateral fingertip-pad contact has not yet occurred, the free block is held at its initial pose. This prevents the first finger from toppling the object. Once both thumb and index fingertip contacts are observed, the object is released on the next control frame, so stable success still has to survive free-body physics.

The v0.4.3 physics-clock synchronization and the WRAP configuration are unchanged.
