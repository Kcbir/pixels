from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .metrics import read_history, summarize_pair
from .output import write_csv, write_json


COLORS = {'unenhanced': '#5d91d4', 'enhanced': '#24a899'}


def save_figure(figure, path: str | Path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(figure)


def plot_stages(stages: dict, path: str | Path):
    figure, axes = plt.subplots(2, len(stages), figsize=(3 * len(stages), 6), squeeze=False)
    for i, (name, image) in enumerate(stages.items()):
        axes[0, i].imshow(np.clip(image / 255.0, 0.0, 1.0))
        axes[0, i].set_title(name.replace('_', ' ').title())
        axes[0, i].axis('off')
        for channel, color in enumerate(('#db6262', '#37a87b', '#568dd1')):
            histogram, edges = np.histogram(image[..., channel], bins=256, range=(0, 256))
            axes[1, i].plot(edges[:-1], histogram / image.shape[0] / image.shape[1], color=color, linewidth=1)
        axes[1, i].set(xlim=(0, 255), xlabel='Intensity', ylabel='Pixel fraction')
    figure.tight_layout()
    save_figure(figure, path)


def summarize_results(root: str | Path, destination: str | Path) -> dict:
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    summaries, rows = {}, []
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for column, model in enumerate(('v5', 'v11')):
        base = read_history(root / 'unenhanced' / model / 'logs/results.csv')
        enh = read_history(root / 'enhanced' / model / 'logs/results.csv')
        summary = summarize_pair(base, enh)
        summaries[model] = summary
        for representation, history in (('unenhanced', base), ('enhanced', enh)):
            axes[0, column].plot(history['epoch'], history['map50_95'], color=COLORS[representation], label=representation.capitalize(), linewidth=1.6)
            row = summary['baseline' if representation == 'unenhanced' else 'enhanced']
            rows.append({'model': model, 'representation': representation, **row})
        axis = axes[1, column]
        epochs, delta = np.array(summary['epochs']), np.array(summary['paired_delta'])
        axis.plot(epochs, delta, color=COLORS['enhanced'], linewidth=1.4)
        axis.fill_between(epochs, 0, delta, where=delta >= 0, color=COLORS['enhanced'], alpha=0.18)
        axis.axhline(0, color='#657080', linewidth=0.8)
        axis.axhline(summary['matched_mean'], color='#d09042', linestyle='--', label=f"Mean {summary['matched_mean']:.3f}")
        axes[0, column].set(title='YOLOv5n' if model == 'v5' else 'YOLO11n', ylabel='Validation mAP@0.50:0.95', ylim=(0, 1))
        axis.set(xlabel='Epoch', ylabel='Enhanced − unenhanced')
        for row_index in (0, 1):
            axes[row_index, column].legend(frameon=False)
            axes[row_index, column].grid(alpha=0.18)
            axes[row_index, column].spines[['top', 'right']].set_visible(False)
    figure.tight_layout()
    save_figure(figure, destination / 'validation_trajectories.png')
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), squeeze=False)
    for column, model in enumerate(('v5', 'v11')):
        for representation in COLORS:
            history = read_history(root / representation / model / 'logs/results.csv')
            for row, metric in enumerate(('precision', 'recall')):
                axes[row, column].plot(history['epoch'], history[metric], color=COLORS[representation], label=representation)
                axes[row, column].set(xlabel='Epoch', ylabel=metric.capitalize(), ylim=(0, 1), title=model)
                axes[row, column].grid(alpha=0.18)
                axes[row, column].legend(frameon=False)
    figure.tight_layout()
    save_figure(figure, destination / 'precision_recall.png')
    write_csv(destination / 'selected_checkpoints.csv', rows)
    write_json(destination / 'paired_summary.json', summaries)
    return summaries


def plot_allocation(candidates, budget: float, path: str | Path):
    figure, axis = plt.subplots(figsize=(8, 5))
    for candidate in candidates:
        axis.scatter(candidate.cost, candidate.utility, color=COLORS['enhanced'] if candidate.cost <= budget else '#9aa2ab')
        axis.annotate(candidate.name, (candidate.cost, candidate.utility), xytext=(5, 5), textcoords='offset points')
    axis.axvline(budget, color='#d09042', linestyle='--', label='Budget')
    axis.set(xlabel='Measured total cost', ylabel='Detection utility')
    axis.legend(frameon=False)
    axis.grid(alpha=0.18)
    save_figure(figure, path)
