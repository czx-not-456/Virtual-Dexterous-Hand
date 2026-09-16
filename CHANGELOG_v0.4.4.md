# v0.4.4

## PINCH geometry calibration

- Raised the PINCH calibration block so its side faces overlap the actual thumb/index precision-pinch workspace.
- The previous block had `size_z=0.056`, `pos_z=0.126`, so its top was `0.182 m`.
- The hand model closes thumb/index around `z≈0.190–0.193 m`; with 8–8.5 mm fingertip pads the index could skim above the top edge and report zero contact.
- New PINCH block: `size=[0.035, 0.010, 0.066]`, `pos=[-0.014, -0.044, 0.136]`. Its bottom remains on the `z=0.070 m` table and its top is `z=0.202 m`.
- WRAP controller and the v0.4.3 physics-clock synchronization are unchanged.
