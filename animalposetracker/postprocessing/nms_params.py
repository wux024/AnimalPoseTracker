"""The NMS thresholds each caller family uses, in one place.

Three recipes exist by design and the values stay exactly as they were -- this
module only gives them names, so a default can be read (and changed, when a
decision is made) in one spot instead of being a literal somewhere in each file:

- ``TRAINING``: validation / evaluation. A near-zero confidence floor, because
  the metrics want every candidate; per-class suppression.
- ``PREDICT``: the offline predictor (``predict``). The training IoU, but only
  candidates above the deploy confidence.
- ``DEPLOY``: the deployment chain (``infer`` CLI / GUI / stream inference).
  A lower IoU and class-agnostic suppression, tuned for the live picture.

Import this module from anywhere: it imports nothing itself.
"""

TRAINING_NMS_CONF = 0.001
TRAINING_NMS_IOU = 0.7
TRAINING_NMS_AGNOSTIC = False
TRAINING_NMS_MAX_DET = 300

PREDICT_NMS_CONF = 0.25
PREDICT_NMS_IOU = 0.7
PREDICT_NMS_AGNOSTIC = False
PREDICT_NMS_MAX_DET = 300

DEPLOY_NMS_CONF = 0.25
DEPLOY_NMS_IOU = 0.45
DEPLOY_NMS_AGNOSTIC = True
DEPLOY_NMS_MAX_DET = 300

# Default confidence of the second-stage detector (the inferencer's top-down stream).
DEPLOY_DETECTOR_CONF = 0.5

__all__ = [
    "TRAINING_NMS_CONF",
    "TRAINING_NMS_IOU",
    "TRAINING_NMS_AGNOSTIC",
    "TRAINING_NMS_MAX_DET",
    "PREDICT_NMS_CONF",
    "PREDICT_NMS_IOU",
    "PREDICT_NMS_AGNOSTIC",
    "PREDICT_NMS_MAX_DET",
    "DEPLOY_NMS_CONF",
    "DEPLOY_NMS_IOU",
    "DEPLOY_NMS_AGNOSTIC",
    "DEPLOY_NMS_MAX_DET",
    "DEPLOY_DETECTOR_CONF",
]
