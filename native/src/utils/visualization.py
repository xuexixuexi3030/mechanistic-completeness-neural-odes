import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def plot_solution_comparison(t_true: np.ndarray, y_true: np.ndarray, t_pred: np.ndarray, y_pred: np.ndarray, component_names: list[str] | None=None, title: str='Solution Comparison', save_path: str | None=None):
    n_components = y_true.shape[1] if y_true.ndim > 1 else 1
    if component_names is None:
        component_names = [f'Component {i}' for i in range(n_components)]
    fig, axes = plt.subplots(n_components, 1, figsize=(10, 3 * n_components), sharex=True)
    if n_components == 1:
        axes = [axes]
    for i, (ax, name) in enumerate(zip(axes, component_names)):
        y_t = y_true[:, i] if y_true.ndim > 1 else y_true
        y_p = y_pred[:, i] if y_pred.ndim > 1 else y_pred
        ax.plot(t_true, y_t, 'k--', label='True', linewidth=1.5)
        ax.plot(t_pred, y_p, 'b-', label='Predicted', linewidth=1.5)
        ax.set_ylabel(name)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time')
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_parameter_convergence(param_history: list[dict], true_values: dict[str, float], title: str='Parameter Convergence', save_path: str | None=None):
    if not param_history:
        return
    epochs = [entry['epoch'] for entry in param_history]
    param_names = [k for k in param_history[0] if k != 'epoch']
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 1, figsize=(10, 3 * n_params), sharex=True)
    if n_params == 1:
        axes = [axes]
    for ax, name in zip(axes, param_names):
        values = [entry[name] for entry in param_history]
        ax.plot(epochs, values, 'b-', linewidth=1.5, label=f'Estimated {name}')
        if name in true_values:
            ax.axhline(y=true_values[name], color='r', linestyle='--', linewidth=1.5, label=f'True {name} = {true_values[name]:.6f}')
        ax.set_ylabel(name)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Epoch')
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_loss_history(history: list[dict], title: str='Training Loss', save_path: str | None=None):
    if not history:
        return
    epochs = [entry['epoch'] for entry in history]
    loss_names = [key for key in history[0] if key != 'epoch' and (not key.startswith('w_')) and (not key.startswith('update_')) and (not key.startswith('backtrack_'))]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(loss_names)))
    for name, color in zip(loss_names, colors):
        values = [entry.get(name, 0) for entry in history]
        ax.semilogy(epochs, values, label=name, color=color, linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_comparison_table(all_metrics: dict[str, dict], metric_names: list[str] | None=None, title: str='Model Comparison', save_path: str | None=None):
    if not all_metrics:
        return
    if metric_names is None:
        metric_names = ['test_mse', 'test_relative_l2']
    experiment_names = list(all_metrics.keys())
    n_metrics = len(metric_names)
    n_experiments = len(experiment_names)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]
    x = np.arange(n_experiments)
    colors = plt.cm.Set2(np.linspace(0, 1, n_experiments))
    for ax, metric in zip(axes, metric_names):
        values = [all_metrics[name].get(metric, 0) for name in experiment_names]
        bars = ax.bar(x, values, color=colors, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(experiment_names, rotation=45, ha='right')
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{val:.2e}', ha='center', va='bottom', fontsize=8)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
