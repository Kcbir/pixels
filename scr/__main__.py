from pathlib import Path
import argparse
import json


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog='pixels', description='Local underwater compensation, detection and computational allocation.')
    commands = root.add_subparsers(dest='command', required=True)
    enhance = commands.add_parser('enhance')
    enhance.add_argument('--input', required=True, type=Path)
    enhance.add_argument('--output', required=True, type=Path)
    enhance.add_argument('--stages', action='store_true')
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--data', required=True, type=Path)
    prepare.add_argument('--output', required=True, type=Path)
    prepare.add_argument('--representation', choices=('unenhanced', 'enhanced'), required=True)
    summary = commands.add_parser('summarize')
    summary.add_argument('--results', type=Path, default=Path('results'))
    summary.add_argument('--output', type=Path, default=Path('runs/summary'))
    allocation = commands.add_parser('allocate')
    allocation.add_argument('--candidates', type=Path, required=True)
    allocation.add_argument('--budget', type=float, required=True)
    allocation.add_argument('--output', type=Path, required=True)
    evaluate = commands.add_parser('metrics')
    evaluate.add_argument('--predictions', type=Path, required=True)
    evaluate.add_argument('--targets', type=Path, required=True)
    evaluate.add_argument('--classes', type=int, required=True)
    evaluate.add_argument('--output', type=Path, required=True)
    for name in ('train', 'predict', 'validate', 'benchmark', 'export'):
        command = commands.add_parser(name)
        command.add_argument('--family', choices=('v5', 'v11'), required=True)
        command.add_argument('--weights', type=Path, required=True)
        command.add_argument('--yolov5-root', type=Path)
        command.add_argument('--device', default='cpu')
        command.add_argument('--image-size', type=int, default=640)
        command.add_argument('--confidence', type=float, default=0.001 if name == 'validate' else 0.25)
        command.add_argument('--iou', type=float, default=0.7)
        command.add_argument('--max-detections', type=int, default=300)
        command.add_argument('--output', type=Path, required=True)
        if name in ('train', 'validate'):
            command.add_argument('--data', type=Path, required=True)
        if name == 'train':
            command.add_argument('--recipe', type=Path, required=True)
            command.add_argument('--epochs', type=int)
        if name == 'validate':
            command.add_argument('--split', choices=('val', 'test'), default='val')
        if name in ('predict', 'benchmark'):
            command.add_argument('--input', type=Path, required=True)
            command.add_argument('--representation', choices=('unenhanced', 'enhanced'), required=True)
        if name == 'benchmark':
            command.add_argument('--warmup', type=int, default=5)
            command.add_argument('--repeats', type=int, default=30)
    return root


def main():
    args = parser().parse_args()
    if args.command == 'enhance':
        from .processing import process_images
        result = process_images(args.input, args.output, args.stages)
    elif args.command == 'prepare':
        from .processing import prepare_dataset
        result = prepare_dataset(args.data, args.output, args.representation)
    elif args.command == 'summarize':
        from .plots import summarize_results
        result = summarize_results(args.results, args.output)
        result = {model: {key: value for key, value in summary.items() if key not in ('epochs', 'paired_delta')} for model, summary in result.items()}
    elif args.command == 'allocate':
        from .mathematics import Candidate, finite_allocation
        from .output import write_json
        from .plots import plot_allocation
        from .processing import new_destination
        candidates = [Candidate(**row) for row in json.loads(args.candidates.read_text())]
        result = finite_allocation(candidates, args.budget)
        root = new_destination(args.output)
        write_json(root / 'allocation.json', result)
        plot_allocation(candidates, args.budget, root / 'allocation.png')
    elif args.command == 'metrics':
        from .metrics import detection_metrics
        from .output import write_json
        result = detection_metrics(json.loads(args.predictions.read_text()), json.loads(args.targets.read_text()), args.classes)
        write_json(args.output, result)
    else:
        from .neural import DetectorConfig, benchmark, export_detector, predict_images, train_detector, validate_detector
        config = DetectorConfig(args.family, args.weights, args.device, args.image_size, args.confidence, args.iou, args.max_detections, args.yolov5_root)
        if args.command == 'train':
            result = train_detector(config, args.data, args.output, args.recipe, args.epochs)
        elif args.command == 'validate':
            result = validate_detector(config, args.data, args.output, args.split)
        elif args.command == 'export':
            result = export_detector(config, args.output)
        elif args.command == 'benchmark':
            result = benchmark(config, args.input, args.output, args.representation == 'enhanced', args.warmup, args.repeats)
        else:
            result = predict_images(config, args.input, args.output, args.representation == 'enhanced')
    if isinstance(result, dict):
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        print(result)


if __name__ == '__main__':
    main()
