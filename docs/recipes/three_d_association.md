# 3D association and rendering

## Index vs epipolar

- `index` pairs the n-th instance across cameras; correct only for one animal
- `epipolar` matches by the Sampson distance, capped by `--max-sampson`
- epipolar needs a calibrated rig; a bad calibration fails every match

## Rendering options

- `--video overlay` projects 3D points back onto each camera
- `--video space` renders a revolving 3D view
- `--images` writes one JPEG per frame
- `--filtered` reads the smoothed sequence instead

## Reading the report

- `association_stats` records match counts per frame
- `view_indices` on each instance shows which detection was chosen
- a falling average view count means the association is failing
