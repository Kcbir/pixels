from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter_ns
import shutil

import numpy as np
import yaml

from .input import image_paths, read_dataset, read_rgb, sha256
from .output import atomic_bytes, environment, write_json, write_rgb
from .stage_1_red import compensate_red
from .stage_2_balance import gray_world
from .stage_3_clahe import equalize_channels
from .stage_4_sharpen import sharpen_channels
from .stage_5_stretch import stretch_channels
from .stage_6_bilateral import bilateral_filter


@dataclass(frozen=True)
class Compensation:
    clip_limit: float = 2.0
    grid: tuple[int, int] = (10, 10)
    alpha: float = 0.3
    diameter: int = 5
    sigma_color: float = 75.0
    sigma_space: float = 75.0

    def __post_init__(self):
        if not np.isfinite([self.clip_limit, self.alpha, self.sigma_color, self.sigma_space]).all():
            raise ValueError('Parameters must be finite')
        if self.clip_limit <= 0 or self.alpha < 0 or self.diameter < 1 or self.diameter % 2 != 1 or min(self.grid) < 1 or min(self.sigma_color, self.sigma_space) <= 0:
            raise ValueError('Invalid compensation parameters')


def compensate(rgb: np.ndarray, parameters: Compensation = Compensation(), capture: bool = False):
    if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) == 0 or not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 255:
        raise ValueError('Expected a finite H x W x 3 RGB array in [0, 255]')
    stages = {'input': rgb.copy()} if capture else {}
    image = rgb
    operations = (
        ('red', lambda x: compensate_red(x)),
        ('balance', lambda x: gray_world(x)),
        ('clahe', lambda x: equalize_channels(x, parameters.clip_limit, parameters.grid)),
        ('sharpen', lambda x: sharpen_channels(x, parameters.alpha)),
        ('stretch', lambda x: stretch_channels(x)),
        ('bilateral', lambda x: bilateral_filter(x, parameters.diameter, parameters.sigma_color, parameters.sigma_space)),
    )
    timings = {}
    for name, operation in operations:
        start = perf_counter_ns()
        image = operation(image)
        timings[name] = (perf_counter_ns() - start) / 1e6
        if capture:
            stages[name] = image.copy()
    return image, stages, timings


def new_destination(path: str | Path, source: str | Path | None = None) -> Path:
    target = Path(path).expanduser().resolve()
    if source is not None:
        origin = Path(source).expanduser().resolve()
        if target == origin or origin in target.parents:
            raise ValueError('Output must be outside the input tree')
    if target.exists():
        raise FileExistsError(f'Choose a new output directory: {target}')
    target.mkdir(parents=True)
    return target


def process_images(source: str | Path, destination: str | Path, capture: bool = False, parameters: Compensation = Compensation()) -> Path:
    paths = image_paths(source)
    base = Path(source).expanduser().resolve()
    root = new_destination(destination, base)
    records = []
    for path in paths:
        relative = path.relative_to(base) if base.is_dir() else Path(path.name)
        output = root / 'images' / relative.with_suffix('.png')
        if output.exists():
            raise ValueError(f'Output name collision: {output}')
        result, stages, timing = compensate(read_rgb(path), parameters, capture)
        write_rgb(output, result)
        if capture:
            from .plots import plot_stages
            for name, image in stages.items():
                write_rgb(root / 'stages' / relative / f'{name}.png', image)
            plot_stages(stages, root / 'plots' / relative.with_suffix('.png'))
        records.append({'source': str(path), 'source_sha256': sha256(path), 'output': str(output.relative_to(root)), 'stage_ms': timing})
    write_json(root / 'manifest.json', {'parameters': asdict(parameters), 'environment': environment(), 'images': records})
    return root


def prepare_dataset(source: str | Path, destination: str | Path, representation: str, parameters: Compensation = Compensation()) -> Path:
    if representation not in {'enhanced', 'unenhanced'}:
        raise ValueError(representation)
    config, samples = read_dataset(source)
    target_path = Path(destination).expanduser().resolve()
    if any(target_path == sample.image.parent or sample.image.parent in target_path.parents for sample in samples):
        raise ValueError('Dataset output must be outside source image directories')
    root = new_destination(destination, Path(source).expanduser().resolve().parent)
    records = []
    for sample in samples:
        relative = sample.relative.with_suffix('.png') if representation == 'enhanced' else sample.relative
        output = root / sample.split / 'images' / relative
        if output.exists():
            raise ValueError(f'Output name collision: {output}')
        timing = {}
        if representation == 'enhanced':
            rgb, _, timing = compensate(read_rgb(sample.image), parameters)
            write_rgb(output, rgb)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sample.image, output)
        label = root / sample.split / 'labels' / relative.with_suffix('.txt')
        if label.exists():
            raise ValueError(f'Label name collision: {label}')
        label.parent.mkdir(parents=True, exist_ok=True)
        if sample.label:
            shutil.copy2(sample.label, label)
        else:
            label.touch()
        records.append({'split': sample.split, 'source': str(sample.image), 'source_sha256': sha256(sample.image), 'image': str(output.relative_to(root)), 'label_sha256': sha256(label), 'missing_source_label': sample.label is None, 'stage_ms': timing})
    descriptor = {'path': str(root), 'names': config['names'], 'nc': len(config['names'])}
    descriptor.update({split: f'{split}/images' for split in ('train', 'val', 'test') if any(s.split == split for s in samples)})
    atomic_bytes(root / 'data.yaml', yaml.safe_dump(descriptor, sort_keys=False).encode())
    write_json(root / 'manifest.json', {'representation': representation, 'parameters': asdict(parameters), 'environment': environment(), 'samples': records})
    return root / 'data.yaml'
