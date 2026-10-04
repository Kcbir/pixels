from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter_ns
import importlib
import os
import shutil
import subprocess
import sys

import numpy as np
import yaml

from .input import image_paths, local_path, read_dataset, read_rgb, sha256
from .output import environment, write_json, write_rgb
from .processing import compensate, new_destination


@dataclass(frozen=True)
class DetectorConfig:
    family: str
    weights: Path
    device: str = 'cpu'
    image_size: int = 640
    confidence: float = 0.25
    iou: float = 0.7
    max_detections: int = 300
    yolov5_root: Path | None = None

    def __post_init__(self):
        if self.family not in {'v5', 'v11'} or self.image_size < 32 or self.max_detections < 1:
            raise ValueError('Invalid detector family, image size, or maximum detections')
        if not 0 <= self.confidence <= 1 or not 0 <= self.iou <= 1:
            raise ValueError('Confidence and IoU thresholds must lie in [0, 1]')
        if ',' in self.device:
            raise ValueError('Use one device per command')
        local_path(self.weights)
        if self.family == 'v5':
            if self.yolov5_root is None:
                raise ValueError('YOLOv5 requires an explicit local upstream checkout')
            local_path(self.yolov5_root, directory=True)


def configure_runtime():
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    os.environ.setdefault('YOLO_OFFLINE', 'true')
    os.environ.setdefault('WANDB_MODE', 'disabled')
    os.environ.setdefault('COMET_MODE', 'DISABLED')


def v5_module(root: Path, name: str):
    root = local_path(root, directory=True)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module(name)
    if root not in Path(module.__file__).resolve().parents:
        raise ImportError(f'{name} was loaded outside the supplied YOLOv5 checkout; use a separate process')
    return module


class Detector:
    def __init__(self, config: DetectorConfig):
        configure_runtime()
        import torch
        self.config = config
        self.torch = torch
        weights = str(local_path(config.weights))
        if config.family == 'v11':
            from ultralytics import YOLO
            self.model = YOLO(weights, task='detect')
        else:
            self.model = torch.hub.load(str(local_path(config.yolov5_root, directory=True)), 'custom', path=weights, source='local', device=config.device, autoshape=True, _verbose=False)
            self.model.conf = config.confidence
            self.model.iou = config.iou
            self.model.max_det = config.max_detections

    def synchronize(self):
        if self.torch.cuda.is_available() and self.config.device not in {'cpu', 'mps'}:
            device = self.config.device if self.config.device.startswith('cuda') else f'cuda:{self.config.device or "0"}'
            self.torch.cuda.synchronize(device)
        if self.config.device == 'mps' and self.torch.backends.mps.is_available():
            self.torch.mps.synchronize()

    def model_information(self) -> dict:
        module = self.model.model if self.config.family == 'v11' else self.model
        return {'parameters': sum(parameter.numel() for parameter in module.parameters()), 'torch_version': self.torch.__version__, 'cuda_version': self.torch.version.cuda, 'cuda_devices': [self.torch.cuda.get_device_name(i) for i in range(self.torch.cuda.device_count())], 'requested_device': self.config.device}

    def predict(self, rgb: np.ndarray) -> tuple[list[dict], np.ndarray]:
        pixels = np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)
        with self.torch.inference_mode():
            if self.config.family == 'v11':
                result = self.model.predict(source=np.ascontiguousarray(pixels[..., ::-1]), imgsz=self.config.image_size, conf=self.config.confidence, iou=self.config.iou, max_det=self.config.max_detections, device=self.config.device, verbose=False, save=False)[0]
                boxes = result.boxes.data.detach().cpu().numpy()
                annotation = np.ascontiguousarray(result.plot()[..., ::-1])
                names = result.names
            else:
                result = self.model(pixels, size=self.config.image_size)
                boxes = result.xyxy[0].detach().cpu().numpy()
                annotation = result.render()[0]
                names = self.model.names
        records = [{'bbox': row[:4].tolist(), 'confidence': float(row[4]), 'class_id': int(row[5]), 'class_name': names[int(row[5])]} for row in boxes]
        return records, annotation


def predict_images(config: DetectorConfig, source: str | Path, destination: str | Path, enhanced: bool) -> Path:
    paths = image_paths(source)
    origin = Path(source).expanduser().resolve()
    root = new_destination(destination, origin)
    detector = Detector(config)
    predictions, timings = [], []
    for path in paths:
        detector.synchronize()
        start = perf_counter_ns()
        rgb = read_rgb(path)
        preprocessing_ms = 0.0
        if enhanced:
            mark = perf_counter_ns()
            rgb, _, _ = compensate(rgb)
            preprocessing_ms = (perf_counter_ns() - mark) / 1e6
        boxes, annotation = detector.predict(rgb)
        detector.synchronize()
        total_ms = (perf_counter_ns() - start) / 1e6
        relative = path.relative_to(origin) if origin.is_dir() else Path(path.name)
        destination_image = root / 'images' / relative.with_suffix('.png')
        if destination_image.exists():
            raise ValueError(f'Output name collision: {destination_image}')
        write_rgb(destination_image, annotation)
        predictions.extend({'image_id': relative.as_posix(), **box} for box in boxes)
        timings.append({'image_id': relative.as_posix(), 'total_ms': total_ms, 'preprocessing_ms': preprocessing_ms})
    write_json(root / 'predictions.json', predictions)
    write_json(root / 'run.json', {'config': asdict(config), 'representation': 'enhanced' if enhanced else 'unenhanced', 'weights_sha256': sha256(local_path(config.weights)), 'environment': environment(), 'timings': timings, 'timing_boundary': 'Decode through rendered predictions; includes compensation, transfers, detector preparation and NMS; excludes output encoding and file writes. Includes cold-start effects.'})
    return root


def benchmark(config: DetectorConfig, source: str | Path, destination: str | Path, enhanced: bool, warmup: int = 5, repeats: int = 30) -> Path:
    if warmup < 0 or repeats < 1:
        raise ValueError('Require nonnegative warmup and positive repeats')
    paths = image_paths(source)
    root = new_destination(destination, Path(source).expanduser().resolve())
    detector = Detector(config)
    def cycle(path):
        detector.synchronize()
        start = perf_counter_ns()
        rgb = read_rgb(path)
        decoded = perf_counter_ns()
        if enhanced:
            rgb, _, _ = compensate(rgb)
        processed = perf_counter_ns()
        detector.predict(rgb)
        detector.synchronize()
        finished = perf_counter_ns()
        return {'decode_ms': (decoded - start) / 1e6, 'preprocessing_ms': (processed - decoded) / 1e6, 'detector_and_render_ms': (finished - processed) / 1e6, 'total_ms': (finished - start) / 1e6}
    for index in range(warmup):
        cycle(paths[index % len(paths)])
    rows = [{'repeat': iteration + 1, 'image': str(path), **cycle(path)} for iteration in range(repeats) for path in paths]
    summaries = {key: {'mean': float(np.mean(values)), 'median': float(np.median(values)), 'p95': float(np.quantile(values, 0.95)), 'minimum': float(np.min(values))} for key in ('decode_ms', 'preprocessing_ms', 'detector_and_render_ms', 'total_ms') for values in [[row[key] for row in rows]]}
    write_json(root / 'latency.json', {'config': asdict(config), 'enhanced': enhanced, 'warmup': warmup, 'repeats': repeats, 'batch_size': 1, 'torch_threads': detector.torch.get_num_threads(), 'model': detector.model_information(), 'environment': environment(), 'summary_ms': summaries, 'samples': rows, 'boundary': 'Local image decoding through CPU materialization and rendering of detections. Includes compensation, resizing, device transfers, model inference and NMS. Excludes model loading and output file writes. OS caches are not cleared.'})
    return root


def train_detector(config: DetectorConfig, data: str | Path, destination: str | Path, recipe: str | Path, epochs: int | None = None) -> Path:
    configure_runtime()
    descriptor = local_path(data)
    read_dataset(descriptor)
    recipe_path = local_path(recipe)
    recorded = yaml.safe_load(recipe_path.read_text())
    if not isinstance(recorded, dict):
        raise ValueError('Recipe must be a mapping')
    root = new_destination(destination)
    write_json(root / 'environment.json', environment())
    shutil.copy2(recipe_path, root / 'source_recipe.yaml')
    if config.family == 'v11':
        from ultralytics import YOLO
        from ultralytics.cfg import DEFAULT_CFG_DICT
        keys = ('epochs', 'patience', 'batch', 'workers', 'optimizer', 'seed', 'deterministic', 'cos_lr', 'close_mosaic', 'amp', 'cache', 'rect', 'single_cls', 'fraction', 'freeze', 'multi_scale', 'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs', 'warmup_momentum', 'warmup_bias_lr', 'box', 'cls', 'dfl', 'nbs', 'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale', 'shear', 'perspective', 'flipud', 'fliplr', 'mosaic', 'mixup', 'copy_paste', 'auto_augment', 'erasing')
        options = {key: recorded[key] for key in keys if key in recorded}
        unsupported = set(options) - set(DEFAULT_CFG_DICT)
        if unsupported:
            raise ValueError(f'The installed Ultralytics version does not support recipe keys: {sorted(unsupported)}')
        options.update(data=str(descriptor), imgsz=config.image_size, device=config.device, project=str(root), name='run', exist_ok=False, save=True, val=True, plots=True)
        if epochs is not None:
            options['epochs'] = epochs
        model = YOLO(str(local_path(config.weights)), task='detect')
        state = {'score': -1.0}
        def select(trainer):
            score = float(trainer.metrics['metrics/mAP50-95(B)'])
            if np.isfinite(score) and score > state['score']:
                state['score'] = score
                shutil.copy2(trainer.last, Path(trainer.wdir) / 'selected_map.pt')
                write_json(root / 'selection.json', {'epoch': trainer.epoch + 1, 'validation_map50_95': score, 'rule': 'Maximum validation mAP50-95; earliest tie'})
        model.add_callback('on_model_save', select)
        write_json(root / 'effective_recipe.json', options)
        model.train(**options)
    else:
        trainer = v5_module(config.yolov5_root, 'train')
        callbacks_type = v5_module(config.yolov5_root, 'utils.callbacks').Callbacks
        callbacks = callbacks_type()
        state = {'score': -1.0, 'selected_epoch': -1}
        def observe(values, epoch, best_fitness, fitness):
            score = float(values[6])
            if np.isfinite(score) and score > state['score']:
                state.update(score=score, selected_epoch=epoch)
        def select(last, epoch, final_epoch, best_fitness, fitness):
            if state['selected_epoch'] == epoch:
                shutil.copy2(last, Path(last).parent / 'selected_map.pt')
                write_json(root / 'selection.json', {'epoch': epoch + 1, 'validation_map50_95': state['score'], 'rule': 'Maximum validation mAP50-95; earliest tie'})
        callbacks.register_action('on_fit_epoch_end', callback=observe)
        callbacks.register_action('on_model_save', callback=select)
        keys = ('epochs', 'batch_size', 'rect', 'cache', 'image_weights', 'multi_scale', 'single_cls', 'optimizer', 'workers', 'quad', 'cos_lr', 'label_smoothing', 'patience', 'freeze', 'seed', 'noautoanchor')
        options = {key: recorded[key] for key in keys if key in recorded}
        hyperparameters = recorded.get('hyp')
        if not isinstance(hyperparameters, dict):
            hyp_path = recipe_path.parent / 'hyp.yaml'
            hyperparameters = yaml.safe_load(local_path(hyp_path).read_text())
        options.update(weights=str(local_path(config.weights)), data=str(descriptor), hyp=hyperparameters, imgsz=config.image_size, device=config.device, project=str(root), name='run', exist_ok=False, nosave=False, noval=False, noplots=False)
        if epochs is not None:
            options['epochs'] = epochs
        write_json(root / 'effective_recipe.json', options)
        revision = subprocess.run(['git', '-C', str(config.yolov5_root), 'rev-parse', 'HEAD'], text=True, capture_output=True)
        write_json(root / 'upstream.json', {'checkout': str(config.yolov5_root), 'revision': revision.stdout.strip() if revision.returncode == 0 else None})
        trainer.run(callbacks=callbacks, **options)
    return root


def validate_detector(config: DetectorConfig, data: str | Path, destination: str | Path, split: str = 'val') -> Path:
    configure_runtime()
    descriptor = local_path(data)
    dataset, _ = read_dataset(descriptor)
    if split not in ('val', 'test') or not dataset.get(split):
        raise ValueError(f'Unavailable evaluation split: {split}')
    root = new_destination(destination)
    if config.family == 'v11':
        from ultralytics import YOLO
        result = YOLO(str(local_path(config.weights)), task='detect').val(data=str(descriptor), split=split, imgsz=config.image_size, batch=16, device=config.device, conf=config.confidence, iou=config.iou, max_det=config.max_detections, project=str(root), name='run', plots=True, save_json=True)
        metrics = {key: float(value) for key, value in result.results_dict.items()}
    else:
        validator = v5_module(config.yolov5_root, 'val')
        if config.max_detections != 300:
            raise ValueError('The YOLOv5 validation adapter uses its native maximum of 300 detections')
        result, maps, timing = validator.run(data=str(descriptor), weights=str(local_path(config.weights)), batch_size=16, imgsz=config.image_size, conf_thres=config.confidence, iou_thres=config.iou, device=config.device, task=split, project=str(root), name='run', plots=True, save_json=True)
        metrics = dict(zip(('precision', 'recall', 'map50', 'map50_95', 'box_loss', 'objectness_loss', 'classification_loss'), map(float, result)))
        metrics['class_map'] = np.asarray(maps).tolist()
        metrics['native_timing_ms'] = list(map(float, timing))
    write_json(root / 'evaluation.json', {'split': split, 'config': asdict(config), 'environment': environment(), 'metrics': metrics})
    return root


def export_detector(config: DetectorConfig, destination: str | Path) -> Path:
    configure_runtime()
    root = new_destination(destination)
    weights = root / 'model.pt'
    shutil.copy2(local_path(config.weights), weights)
    if config.family == 'v11':
        from ultralytics import YOLO
        exported = YOLO(str(weights), task='detect').export(format='onnx', imgsz=config.image_size, device=config.device, dynamic=False, simplify=False)
        target = Path(exported)
    else:
        command = [sys.executable, str(local_path(config.yolov5_root / 'export.py')), '--weights', str(weights), '--include', 'onnx', '--imgsz', str(config.image_size), '--device', config.device]
        subprocess.run(command, cwd=config.yolov5_root, check=True)
        target = weights.with_suffix('.onnx')
    if not target.exists():
        raise RuntimeError('The exporter did not produce an ONNX file')
    write_json(root / 'export.json', {'source_sha256': sha256(local_path(config.weights)), 'onnx_sha256': sha256(target), 'config': asdict(config), 'environment': environment()})
    return target
