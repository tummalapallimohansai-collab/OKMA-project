

\# Detection Confidence Thresholds



The OKMA system uses different confidence thresholds for helmet and fall detection depending on the inference mode.



| Inference File | Detection Type | Confidence Threshold |

|---|---|---:|

| testing/live\_detection.py | Helmet | 0.50 |

| testing/live\_detection.py | Fall | 0.40 |

| testing/live\_monitoring.py | Helmet | 0.70 |

| testing/live\_monitoring.py | Fall | 0.50 |



\*\*Note:\*\* These values are taken from the current inference source code. They are confidence thresholds used during prediction, not independently optimized validation thresholds.



