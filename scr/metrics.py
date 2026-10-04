from pathlib import Path
import csv

import numpy as np


ALIASES = {
    'precision': ('metrics/precision', 'metrics/precision(B)'),
    'recall': ('metrics/recall', 'metrics/recall(B)'),
    'map50': ('metrics/mAP_0.5', 'metrics/mAP50(B)'),
    'map50_95': ('metrics/mAP_0.5:0.95', 'metrics/mAP50-95(B)'),
}


def read_history(path: str | Path) -> dict[str, np.ndarray]:
    with Path(path).open(newline='') as stream:
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(stream)]
    if not rows:
        raise ValueError(f'Empty history: {path}')
    result = {'epoch': np.array([int(float(row['epoch'])) for row in rows])}
    if result['epoch'][0] == 0:
        result['epoch'] += 1
    if np.any(np.diff(result['epoch']) != 1):
        raise ValueError('Histories require unique consecutive epochs')
    for name, aliases in ALIASES.items():
        key = next((key for key in aliases if key in rows[0]), None)
        if key is None:
            raise ValueError(f'Missing {name} in {path}')
        result[name] = np.array([float(row[key]) for row in rows])
        if not np.isfinite(result[name]).all():
            raise ValueError(f'Nonfinite {name} in {path}')
    return result


def summarize_run(history: dict) -> dict:
    values = history['map50_95']
    best = int(np.argmax(values))
    p, r = float(history['precision'][best]), float(history['recall'][best])
    return {'epochs': len(values), 'selected_epoch': int(history['epoch'][best]), **{key: float(history[key][best]) for key in ALIASES}, 'f1': 2 * p * r / (p + r) if p + r else 0.0, 'final10': float(values[-10:].mean()), 'top5': float(np.sort(values)[-5:].mean()), 'final25': float(values[-25:].mean())}


def summarize_pair(baseline: dict, enhanced: dict) -> dict:
    base, enh = summarize_run(baseline), summarize_run(enhanced)
    epochs, i, j = np.intersect1d(baseline['epoch'], enhanced['epoch'], return_indices=True)
    if not len(epochs):
        raise ValueError('Runs have no shared epochs')
    delta = enhanced['map50_95'][j] - baseline['map50_95'][i]
    return {'baseline': base, 'enhanced': enh, 'delta_best': enh['map50_95'] - base['map50_95'], 'delta_final10': enh['final10'] - base['final10'], 'delta_top5': enh['top5'] - base['top5'], 'matched_mean': float(delta.mean()), 'wins': int((delta > 0).sum()), 'shared_epochs': len(epochs), 'win_fraction': float((delta > 0).mean()), 'epochs': epochs.tolist(), 'paired_delta': delta.tolist()}


def box_iou(boxes_a, boxes_b) -> np.ndarray:
    a, b = np.asarray(boxes_a, dtype=float).reshape(-1, 4), np.asarray(boxes_b, dtype=float).reshape(-1, 4)
    intersection = np.maximum(0.0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2])).prod(axis=-1)
    area_a = np.maximum(0.0, a[:, 2:] - a[:, :2]).prod(axis=-1)
    area_b = np.maximum(0.0, b[:, 2:] - b[:, :2]).prod(axis=-1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def average_precision(recall, precision) -> float:
    r = np.concatenate(([0.0], recall, [1.0]))
    p = np.concatenate(([1.0], precision, [0.0]))
    p = np.maximum.accumulate(p[::-1])[::-1]
    grid = np.linspace(0.0, 1.0, 101)
    return float(np.trapezoid(np.interp(grid, r, p), grid))


def detection_metrics(predictions: list[dict], targets: list[dict], classes: int) -> dict:
    if classes < 1:
        raise ValueError('Class count must be positive')
    for record in predictions + targets:
        box = np.asarray(record['bbox'], dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]) or int(record['class_id']) != record['class_id'] or not 0 <= record['class_id'] < classes:
            raise ValueError('Expected class IDs and nondegenerate finite xyxy boxes')
    for record in predictions:
        if not np.isfinite(record['confidence']) or not 0 <= record['confidence'] <= 1:
            raise ValueError('Confidence must lie in [0, 1]')
    thresholds = np.linspace(0.5, 0.95, 10)
    per_class = []
    for class_id in range(classes):
        references = [r for r in targets if r['class_id'] == class_id]
        if not references:
            continue
        ranked = sorted((r for r in predictions if r['class_id'] == class_id), key=lambda r: -r['confidence'])
        aps = []
        curves = []
        for threshold in thresholds:
            used = set()
            true_positive = np.zeros(len(ranked))
            for index, prediction in enumerate(ranked):
                eligible = [(j, ref) for j, ref in enumerate(references) if ref['image_id'] == prediction['image_id'] and j not in used]
                if eligible:
                    overlaps = box_iou([prediction['bbox']], [ref['bbox'] for _, ref in eligible])[0]
                    winner = int(overlaps.argmax())
                    if overlaps[winner] >= threshold:
                        used.add(eligible[winner][0])
                        true_positive[index] = 1.0
            tp = np.cumsum(true_positive)
            recall = tp / len(references)
            precision = tp / np.arange(1, len(ranked) + 1)
            aps.append(average_precision(recall, precision) if len(ranked) else 0.0)
            curves.append({'recall': recall.tolist(), 'precision': precision.tolist()})
        per_class.append({'class_id': class_id, 'instances': len(references), 'ap': aps, 'curves': curves})
    if not per_class:
        raise ValueError('At least one annotated target is required')
    ap = np.array([row['ap'] for row in per_class])
    return {'map50': float(ap[:, 0].mean()), 'map50_95': float(ap.mean()), 'thresholds': thresholds.tolist(), 'per_class': per_class, 'protocol': 'Confidence-ordered one-to-one matching; 101-point interpolated trapezoidal AP; no crowd or ignore annotations. Use native framework validation for archive comparisons.'}
