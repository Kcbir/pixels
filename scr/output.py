from dataclasses import asdict, is_dataclass
from pathlib import Path
import csv
import json
import os
import platform
import tempfile
from importlib.metadata import PackageNotFoundError, version

import cv2
import numpy as np


def json_value(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def atomic_bytes(path: str | Path, data: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=target.parent, prefix='.pixels-')
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def write_json(path: str | Path, value) -> Path:
    return atomic_bytes(path, (json.dumps(value, default=json_value, indent=2, allow_nan=False) + '\n').encode())


def write_rgb(path: str | Path, rgb: np.ndarray) -> Path:
    target = Path(path)
    pixels = np.rint(np.clip(rgb, 0.0, 255.0)).astype(np.uint8)
    success, encoded = cv2.imencode(target.suffix, cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
    if not success:
        raise OSError(f'Unable to encode {target}')
    return atomic_bytes(target, encoded.tobytes())


def write_csv(path: str | Path, rows: list[dict]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('Cannot write an empty table without a schema')
    with target.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    return target


def environment() -> dict:
    packages = {}
    for name in ('numpy', 'opencv-python', 'scipy', 'matplotlib', 'torch', 'ultralytics', 'PyYAML'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {'python': platform.python_version(), 'platform': platform.platform(), 'packages': packages}
