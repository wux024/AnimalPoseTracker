# The records file

Every inference run writes `results/inference/records.json`. It is the run's result: the video
is a rendering of it, not the other way round.

## Schema

```json
[
  {
    "frame": 0,
    "predictions": [
      {
        "class": 0,
        "score": 0.93,
        "box": [120.0, 88.0, 310.0, 260.0],
        "track_id": 1,
        "keypoints": [[210.0, 150.0, 0.91], [214.0, 132.0, 0.88]]
      }
    ]
  }
]
```

| Field | Meaning |
|---|---|
| `frame` | frame index in the source, starting at 0 |
| `predictions` | one entry per instance in that frame |
| `class` | class index into the configured `names` |
| `score` | detection confidence |
| `box` | `[x1, y1, x2, y2]` in source pixels |
| `track_id` | present only when tracking was on; stable while the track survives |
| `keypoints` | `[[x, y, score], ...]`, one per configured keypoint |

## Tracking ids

`track_id` is assigned by the tracker, not the model. Two consequences:

- a track that is lost and re-acquired gets a new id, so the same animal can appear twice in a
  long sequence;
- replaying the records with another tracker renumbers them — the ids mean nothing outside the
  run that produced them.

## Reading records

```python
import json

records = json.load(open("results/inference/records.json"))
for entry in records:
    for instance in entry["predictions"]:
        points = instance["keypoints"]
```

The shape is the same one `predict` writes for `predictions.json`, which is why the tracking and
filtering stages can read either: they take records, not frames.

## Filtering the records

`--pose-filter` writes a sibling file with the smoothed keypoints, leaving `records.json`
untouched. That is the point of keeping records as data: a filter is a lens over them, not a
destructive pass.
