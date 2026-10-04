from dataclasses import dataclass
from pathlib import Path
import hashlib

import cv2
import numpy as np
import yaml


IMAGE_SUFFIXES = frozenset({'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'})


@dataclass(frozen=True)
class Sample:
    image: Path
    label: Path | None
    split: str
    relative: Path


def local_path(value: str | Path, directory: bool = False) -> Path:
    path = Path(value).expanduser().resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise FileNotFoundError(path)
    return path


def read_rgb(path: str | Path) -> np.ndarray:
    source = local_path(path)
    decoded = cv2.imdecode(np.fromfile(source, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 3 or decoded.dtype != np.uint8:
        raise ValueError(f'Expected an 8-bit three-channel image: {source}')
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def image_paths(source: str | Path) -> list[Path]:
    path = Path(source).expanduser().resolve()
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    paths = sorted(p for p in path.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise ValueError(f'No supported images in {path}')
    return paths


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_dataset(path: str | Path) -> tuple[dict, list[Sample]]:
    descriptor = local_path(path)
    config = yaml.safe_load(descriptor.read_text())
    if not isinstance(config, dict) or not config.get('names'):
        raise ValueError('Dataset YAML must define class names and local train/val paths')
    names = config['names']
    if isinstance(names, dict) and sorted(names) != list(range(len(names))):
        raise ValueError('Class IDs must be contiguous integers beginning at zero')
    root = Path(config.get('path', descriptor.parent)).expanduser()
    if not root.is_absolute():
        root = descriptor.parent / root
    root = root.resolve()
    samples = []
    identities = {}
    for split in ('train', 'val', 'test'):
        entries = config.get(split)
        if not entries:
            if split != 'test':
                raise ValueError(f'Missing {split} split')
            continue
        if isinstance(entries, str):
            entries = [entries]
        for entry in entries:
            location = Path(entry).expanduser()
            location = (location if location.is_absolute() else root / location).resolve()
            if location.suffix.lower() == '.txt':
                paths = []
                for line in local_path(location).read_text().splitlines():
                    if line.strip():
                        item = Path(line.strip()).expanduser()
                        paths.append(local_path(item if item.is_absolute() else location.parent / item))
                anchor = root
            else:
                paths = image_paths(location)
                anchor = location if location.is_dir() else location.parent
            for image in paths:
                if image in identities:
                    raise ValueError(f'Image occurs more than once: {image} ({identities[image]}, {split})')
                identities[image] = split
                parts = list(image.parts)
                positions = [i for i, part in enumerate(parts) if part == 'images']
                if not positions:
                    raise ValueError(f'YOLO dataset requires an images directory: {image}')
                parts[positions[-1]] = 'labels'
                label = Path(*parts).with_suffix('.txt')
                if label.exists():
                    for line in label.read_text().splitlines():
                        if not line.strip():
                            continue
                        fields = [float(x) for x in line.split()]
                        if len(fields) != 5 or not np.isfinite(fields).all():
                            raise ValueError(f'Expected finite YOLO detection labels: {label}')
                        cls, x, y, width, height = fields
                        if cls != int(cls) or not 0 <= cls < len(names) or not all(0 <= v <= 1 for v in (x, y, width, height)) or min(width, height) <= 0:
                            raise ValueError(f'Invalid class or normalized box: {label}')
                try:
                    relative = image.relative_to(anchor)
                except ValueError:
                    relative = Path(sha256(image)[:16]) / image.name
                samples.append(Sample(image, label if label.exists() else None, split, relative))
    return config, samples
