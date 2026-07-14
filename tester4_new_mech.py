import argparse
import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from train3_mech import IN_DIM, ROLE_DIM, Regressor43D

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='*.pt checkpoint from train3.py')
    parser.add_argument('--npy', required=True, nargs='+', help='one or more csv files')
    parser.add_argument('--batch_size', type=int, default=65536, help='inference batch size')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--segment_min_out', default='segment_min_samples_new.csv')
    parser.add_argument('--actual4800_min_error_out', default='actual_ge4800_min_error50_new.csv')
    parser.add_argument('--pred4800_max_error_out', default='pred_ge4800_max_error50_new.csv')
    parser.add_argument('--extreme_topk', type=int, default=50)
    parser.add_argument('--segment_dist_out', default='segment_dist_mech.png')
    parser.add_argument('--error_bin_width', type=float, default=25.0, help='bin width for pred-actual error histograms')
    parser.add_argument('--show_scatter', action='store_true', help='draw the full prediction error scatter plot')
    parser.add_argument('--no_plot', action='store_true')
    return parser.parse_args()


ATTR_NAMES = ['血量', '攻', '防', '速', '敏', '魔', '抗', '智']
SKILL_NAMES = [
    '火球术', '冰冻术', '雷击术', '地裂术', '吸血攻击', '投毒', '连击',
    '会心一击', '瘟疫', '生命之轮', '狂暴术', '魅惑', '加速术', '减速术',
    '诅咒', '治愈魔法', '苏生术', '净化', '铁壁', '蓄力', '聚气',
    '潜行', '血祭', '分身', '幻术', '防御', '守护', '伤害反弹',
    '护身符', '护盾', '反击', '吞噬', '召唤亡灵', '垂死抗争', '隐匿',
]
PHANTOM_NAMES = [
    '幻影_攻',
    '幻影_防',
    '幻影_速',
    '幻影_敏',
    '幻影_魔',
    '幻影_抗',
    '幻影_智',
    '幻影_血量',
    '幻影_唯一技能附体',
]
MECH_NAMES = [
    '末尾主动_技能id加1',
    '末尾主动_原始熟练度',
    '末尾主动_受分身衰减机制影响',
    '末尾座位14_技能id加1',
    '末尾座位14_原始熟练度',
    '末尾座位14_bonus',
    '末尾座位14_受分身衰减机制影响',
    '末尾座位15_技能id加1',
    '末尾座位15_原始熟练度',
    '末尾座位15_bonus',
    '末尾座位15_受分身衰减机制影响',
]


def local_feature_name(local_idx):
    if 0 <= local_idx < 8:
        return ATTR_NAMES[local_idx]
    if 8 <= local_idx <= 42:
        skill_idx = local_idx - 8
        if local_idx == 26:
            return 'skill_无盾铁壁'
        return f'skill_{SKILL_NAMES[skill_idx]}'
    if local_idx == 43:
        return '隐匿判定'
    if local_idx == 44:
        return 'skill_有盾铁壁'
    if 45 <= local_idx <= 53:
        return PHANTOM_NAMES[local_idx - 45]
    if 54 <= local_idx <= 64:
        return MECH_NAMES[local_idx - 54]
    return f'unknown_{local_idx}'


def feature_headers():
    headers = []
    for global_idx in range(IN_DIM):
        role = global_idx // ROLE_DIM + 1
        local_idx = global_idx % ROLE_DIM
        headers.append(f'f{global_idx:03d}_p{role}_{local_feature_name(local_idx)}')
    return headers


def load_checkpoint(model_path, device):
    ckpt = torch.load(model_path, map_location=device)
    model = Regressor43D().to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    y_scale = float(ckpt['y_scale'])
    y_bias = float(ckpt.get('y_bias', 0.0))
    x_mean = ckpt.get('x_mean')
    x_std = ckpt.get('x_std')
    if x_mean is not None and not torch.is_tensor(x_mean):
        x_mean = torch.tensor(x_mean, dtype=torch.float32)
    if x_std is not None and not torch.is_tensor(x_std):
        x_std = torch.tensor(x_std, dtype=torch.float32)
    return model, y_scale, y_bias, x_mean, x_std


def load_csvs(paths):
    blocks = []
    source_files = []
    source_rows = []
    for csv_path in paths:
        arr = np.loadtxt(csv_path, delimiter=',', dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] < IN_DIM + 1:
            raise ValueError(f'{csv_path} has {arr.shape[1]} columns, need >= {IN_DIM + 1}')
        arr = arr[:, :IN_DIM + 1]
        blocks.append(arr)
        n_row = arr.shape[0]
        source_files.extend([csv_path] * n_row)
        source_rows.extend(range(1, n_row + 1))
        print(f'[INFO] Loaded {csv_path}: {n_row} rows')

    data = np.concatenate(blocks, axis=0)
    return data[:, 0], data[:, 1:], source_files, source_rows


def infer(model, x_cpu, batch_size, device, x_mean, x_std):
    pred_norms = []
    with torch.no_grad():
        for i in range(0, len(x_cpu), batch_size):
            batch_x = torch.from_numpy(x_cpu[i:i + batch_size]).to(device)
            if x_mean is not None and x_std is not None:
                batch_x = (batch_x - x_mean.to(device)) / x_std.to(device)
            pred_norms.append(model(batch_x).view(-1).cpu().numpy())
    return np.concatenate(pred_norms)


def print_threshold_summary(truth, pred):
    print('\n' + '=' * 22 + ' Threshold Screening ' + '=' * 22)
    print(f"{'target':>8} | {'actual':>8} | {'safe_pred_min':>13} | {'selected':>9} | {'ratio':>8} | {'recall':>8}")
    print('-' * 72)
    for th in [4600, 4700, 4800, 4900, 5000, 5100]:
        mask = truth >= th
        actual = int(mask.sum())
        if actual == 0:
            continue
        safe_min = float(pred[mask].min())
        selected = int((pred >= safe_min).sum())
        recall = float(((pred >= safe_min) & mask).sum() / actual)
        ratio = selected / actual
        print(f'{th:>8.0f} | {actual:>8} | {safe_min:>13.1f} | {selected:>9} | {ratio:>7.2f}x | {recall:>7.3f}')

    print('\n' + '=' * 22 + ' Fixed Prediction Cutoffs ' + '=' * 21)
    print(f"{'pred_cut':>8} | {'selected':>9} | {'actual>=4800':>12} | {'actual>=4900':>12} | {'p10_truth':>9}")
    print('-' * 72)
    for cut in [4600, 4700, 4750, 4800, 4850, 4900, 4950, 5000]:
        mask = pred >= cut
        selected = int(mask.sum())
        if selected == 0:
            continue
        hit4800 = int((truth[mask] >= 4800).sum())
        hit4900 = int((truth[mask] >= 4900).sum())
        p10 = float(np.percentile(truth[mask], 10))
        print(f'{cut:>8.0f} | {selected:>9} | {hit4800:>12} | {hit4900:>12} | {p10:>9.1f}')


def plot_segment_distributions(truth, pred, out_path, no_show, bin_width=100.0):
    segments = [
        (4800, 4900, '4800~4900'),
        (4900, 5000, '4900~5000'),
        (5000, 5100, '5000~5100'),
        (5100, None, '>=5100'),
    ]

    segment_errors = []
    for base_values in (truth, pred):
        for lo, hi, _ in segments:
            mask = base_values >= lo if hi is None else (base_values >= lo) & (base_values < hi)
            segment_errors.append(pred[mask] - truth[mask])

    non_empty = [e for e in segment_errors if e.size > 0]
    bin_width = max(float(bin_width), 1.0)
    if non_empty:
        all_err = np.concatenate(non_empty)
        lo_edge = np.floor(all_err.min() / bin_width) * bin_width
        hi_edge = np.ceil(all_err.max() / bin_width) * bin_width
        lo_edge = min(lo_edge, -50.0)
        hi_edge = max(hi_edge, 50.0)
        bins = np.arange(lo_edge, hi_edge + bin_width, bin_width)
    else:
        bins = np.arange(-200.0, 200.0 + bin_width, bin_width)

    fig, axes = plt.subplots(2, len(segments), figsize=(6 * len(segments), 10), dpi=120)

    def draw_error_hist(ax, err, title, color):
        if err.size > 0:
            ax.hist(err, bins=bins, alpha=0.78, color=color)
            ax.axvline(0, color='#333333', linewidth=1.1)
            ax.axvline(err.mean(), color='red', linestyle='--', linewidth=1.5, label=f'均值 {err.mean():+.1f}')
            ax.axvline(np.median(err), color='green', linestyle=':', linewidth=1.5, label=f'中位数 {np.median(err):+.1f}')
            ax.legend(fontsize=9)
            ax.text(
                0.02,
                0.96,
                f'MAE {np.abs(err).mean():.1f}\nP10 {np.percentile(err, 10):+.1f}\nP90 {np.percentile(err, 90):+.1f}',
                ha='left',
                va='top',
                transform=ax.transAxes,
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.72, edgecolor='none'),
            )
        else:
            ax.text(0.5, 0.5, '无样本', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel('预测偏差 pred - actual')
        ax.set_ylabel('样本数')
        ax.grid(alpha=0.25)

    for col, (lo, hi, label) in enumerate(segments):
        mask_actual = truth >= lo if hi is None else (truth >= lo) & (truth < hi)
        err_actual = pred[mask_actual] - truth[mask_actual]
        draw_error_hist(axes[0, col], err_actual, f'实际{label} 的偏差分布\nN={err_actual.size}', '#4c78a8')

        mask_pred = pred >= lo if hi is None else (pred >= lo) & (pred < hi)
        err_pred = pred[mask_pred] - truth[mask_pred]
        draw_error_hist(axes[1, col], err_pred, f'预测{label} 的偏差分布\nN={err_pred.size}', '#f28e2b')

        if err_actual.size > 0:
            print(f'[ACT {label}] bias={err_actual.mean():+.2f}, MAE={np.abs(err_actual).mean():.2f}, p10={np.percentile(err_actual,10):+.2f}, p90={np.percentile(err_actual,90):+.2f}, N={err_actual.size}')
        else:
            print(f'[ACT {label}] no samples')
        if err_pred.size > 0:
            print(f'[PRED {label}] bias={err_pred.mean():+.2f}, MAE={np.abs(err_pred).mean():.2f}, p10={np.percentile(err_pred,10):+.2f}, p90={np.percentile(err_pred,90):+.2f}, N={err_pred.size}')
        else:
            print(f'[PRED {label}] no samples')

    fig.suptitle('高分段预测偏差分布（pred - actual）', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path)
    print(f'[INFO] Segment error histogram saved to: {out_path}')
    if not no_show:
        plt.show()
    plt.close(fig)


def export_segment_min(truth, pred, x_cpu, source_files, source_rows, out_path):
    segment_min_records = []
    bins = np.arange(4600, int(truth.max()) + 50, 50)
    for b_start in bins:
        mask_bin = truth >= b_start
        if not np.any(mask_bin):
            continue
        bin_indices = np.where(mask_bin)[0]
        global_min_idx = int(bin_indices[np.argmin(pred[mask_bin])])
        segment_min_records.append({
            'segment_start': int(b_start),
            'segment_label': f'>={int(b_start)}',
            'global_index': global_min_idx,
            'source_file': source_files[global_min_idx],
            'source_row': source_rows[global_min_idx],
            'actual_score': float(truth[global_min_idx]),
            'pred_score': float(pred[global_min_idx]),
            'error': float(pred[global_min_idx] - truth[global_min_idx]),
            'features': x_cpu[global_min_idx].astype(np.float32).tolist(),
        })

    headers = [
        'segment_start', 'segment_label', 'global_index',
        'source_file', 'source_row',
        'actual_score', 'pred_score', 'error',
    ] + feature_headers()
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for rec in segment_min_records:
            writer.writerow([
                rec['segment_start'], rec['segment_label'], rec['global_index'],
                rec['source_file'], rec['source_row'],
                rec['actual_score'], rec['pred_score'], rec['error'],
                *rec['features'],
            ])
    print(f'[INFO] Segment min samples saved to: {out_path}')


def export_ranked_samples(indices, truth, pred, err, x_cpu, source_files, source_rows, out_path):
    headers = [
        'rank', 'global_index', 'source_file', 'source_row',
        'actual_score', 'pred_score', 'error',
    ] + feature_headers()
    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for rank, idx in enumerate(indices, start=1):
            idx = int(idx)
            writer.writerow([
                rank, idx, source_files[idx], source_rows[idx],
                float(truth[idx]), float(pred[idx]), float(err[idx]),
                *x_cpu[idx].astype(np.float32).tolist(),
            ])
    print(f'[INFO] Ranked samples saved to: {out_path}')


def export_extreme_error_samples(
    truth,
    pred,
    err,
    x_cpu,
    source_files,
    source_rows,
    topk,
    actual4800_min_error_out,
    pred4800_max_error_out,
):
    actual_mask = truth >= 4800
    if np.any(actual_mask):
        actual_indices = np.where(actual_mask)[0]
        selected = actual_indices[np.argsort(err[actual_mask])[:topk]]
        export_ranked_samples(
            selected,
            truth,
            pred,
            err,
            x_cpu,
            source_files,
            source_rows,
            actual4800_min_error_out,
        )
    else:
        print('[WARN] No samples with actual>=4800; skipped actual4800 min-error export')

    pred_mask = pred >= 4800
    if np.any(pred_mask):
        pred_indices = np.where(pred_mask)[0]
        selected = pred_indices[np.argsort(err[pred_mask])[::-1][:topk]]
        export_ranked_samples(
            selected,
            truth,
            pred,
            err,
            x_cpu,
            source_files,
            source_rows,
            pred4800_max_error_out,
        )
    else:
        print('[WARN] No samples with pred>=4800; skipped pred4800 max-error export')


def main():
    t_total_start = time.perf_counter()
    args = get_args()
    device = args.device
    print(f'[INFO] Using device: {device}')

    model, y_scale, y_bias, x_mean, x_std = load_checkpoint(args.model, device)
    truth, x_cpu, source_files, source_rows = load_csvs(args.npy)

    t0 = time.perf_counter()
    pred_norm = infer(model, x_cpu, args.batch_size, device, x_mean, x_std)
    pred = pred_norm * y_scale + y_bias
    t_infer = time.perf_counter() - t0

    err = pred - truth
    abs_err = np.abs(err)
    print(f'\n[Summary] samples={len(truth)}')
    print(f'MAE={abs_err.mean():.2f}, RMSE={np.sqrt(np.mean(err ** 2)):.2f}, MaxErr={err.max():.1f}, MinErr={err.min():.1f}')
    for th in [4800, 4900, 5000]:
        mask = truth >= th
        if np.any(mask):
            e = err[mask]
            print(f'Actual>={th}: N={mask.sum()}, MAE={np.abs(e).mean():.2f}, min_err={e.min():.1f}, p10_err={np.percentile(e, 10):.1f}')
    low_mask = truth <= 4300
    if np.any(low_mask):
        low_err = err[low_mask]
        print(f'Actual<=4300: N={low_mask.sum()}, over_rate={(low_err > 0).mean():.3f}, p95_over={np.percentile(low_err, 95):.1f}')

    print_threshold_summary(truth, pred)
    export_segment_min(truth, pred, x_cpu, source_files, source_rows, args.segment_min_out)
    export_extreme_error_samples(
        truth,
        pred,
        err,
        x_cpu,
        source_files,
        source_rows,
        args.extreme_topk,
        args.actual4800_min_error_out,
        args.pred4800_max_error_out,
    )
    plot_segment_distributions(truth, pred, args.segment_dist_out, args.no_plot, args.error_bin_width)

    print(f'Inference time: {t_infer * 1000:.2f} ms ({t_infer / len(truth) * 1e6:.2f} us/sample)')
    print(f'Total time: {(time.perf_counter() - t_total_start):.2f} s')

    if args.show_scatter and not args.no_plot:
        plt.figure(figsize=(8, 5))
        plt.scatter(truth, err, s=2, alpha=0.3, c=abs_err, cmap='viridis')
        plt.colorbar(label='Abs Error')
        plt.axhline(0, color='red', linestyle='--', linewidth=1)
        plt.xlabel('Ground Truth')
        plt.ylabel('Error (Pred - Truth)')
        plt.title('Prediction Error Distribution')
        plt.grid(True, alpha=0.3)
        plt.show()
    elif args.show_scatter and args.no_plot:
        print('[WARN] --show_scatter ignored because --no_plot is set')


if __name__ == '__main__':
    main()
