from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_coco(gt_json_path: str | Path, predictions: list[dict]) -> dict:
    gt = COCO(str(gt_json_path))
    if not predictions:
        return {"mAP": 0.0, "mAP50": 0.0, "mAP75": 0.0, "mAR100": 0.0}

    pred = gt.loadRes(predictions)
    evaluator = COCOeval(gt, pred, iouType="bbox")
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        "mAP": float(evaluator.stats[0]),
        "mAP50": float(evaluator.stats[1]),
        "mAP75": float(evaluator.stats[2]),
        "mAR100": float(evaluator.stats[8]),
    }
