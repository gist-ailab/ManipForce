import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

os.environ.setdefault('QT_QPA_PLATFORM_PLUGIN_PATH', '/home/minhwan/anaconda3/envs/umi/plugins')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

parser = argparse.ArgumentParser()
parser.add_argument('dirs', nargs='+', help='eval result 폴더 (각 폴더에 result.csv 필요)')
parser.add_argument('-o', '--output', default=None, help='출력 png 경로')
args = parser.parse_args()

dfs = []
for d in args.dirs:
    csv_path = os.path.join(d, 'result.csv')
    if not os.path.exists(csv_path):
        print(f"[SKIP] {csv_path} 없음")
        continue
    dfs.append(pd.read_csv(csv_path))
    print(f"[LOAD] {csv_path} ({len(dfs[-1])} trials)")
if not dfs:
    print("result.csv를 찾을 수 없습니다.")
    exit(1)

df = pd.concat(dfs, ignore_index=True)
df['mode'] = df['mode_label'].apply(lambda x: 'Baseline' if 'baseline' in x.lower() else 'C2F')

out_path = args.output or os.path.join(os.path.dirname(os.path.commonpath(args.dirs)), 'result_comparison.png')
succ = df[df['result'] == 'SUCCESS'].copy()
succ['task_time_s'] = succ['reaching_time_s'] + succ['contact_time_s']
succ['task_steps'] = succ['reaching_steps'] + succ['contact_steps']
colors = {'Baseline': '#4C72B0', 'C2F': '#DD8452'}

def get_clean_trials(mode, metric='task_time_s'):
    """total_time 기준 IQR로 이상치 trial 제거, 동일 trial set 반환"""
    sub = succ[succ['mode'] == mode]
    q1, q3 = sub[metric].quantile(0.25), sub[metric].quantile(0.75)
    iqr = q3 - q1
    mask = (sub[metric] >= q1 - 1.5 * iqr) & (sub[metric] <= q3 + 1.5 * iqr)
    return sub[mask]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# ── (0,0) Success Rate ──
ax = axes[0, 0]
for i, mode in enumerate(['Baseline', 'C2F']):
    sub = df[df['mode'] == mode]
    total = len(sub)
    success = (sub['result'] == 'SUCCESS').sum()
    rate = success / total * 100
    ax.bar(i, rate, color=colors[mode], width=0.5)
    ax.text(i, rate + 1.5, f'{rate:.0f}%\n({success}/{total})', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.set_xticks([0, 1]); ax.set_xticklabels(['Baseline', 'C2F'])
ax.set_ylabel('Success Rate (%)'); ax.set_ylim(0, 115)
ax.set_title('Success Rate')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ── Helper: stacked bar (동일 trial set) ──
def plot_stacked(ax, metric_reach, metric_contact, unit, title):
    for i, mode in enumerate(['Baseline', 'C2F']):
        clean = get_clean_trials(mode)
        r_mean = clean[metric_reach].mean()
        c_mean = clean[metric_contact].mean()
        total_mean = r_mean + c_mean
        n = len(clean)
        ax.bar(i, r_mean, color=colors[mode], width=0.5, alpha=0.6)
        ax.bar(i, c_mean, bottom=r_mean, color=colors[mode], width=0.5, alpha=1.0)
        ax.text(i, total_mean + 0.5, f'{total_mean:.1f}{unit} (n={n})', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.text(i, r_mean / 2, f'Reach\n{r_mean:.1f}{unit}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        ax.text(i, r_mean + c_mean / 2, f'Contact\n{c_mean:.1f}{unit}', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Baseline', 'C2F'])
    ax.set_title(title)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ── Helper: box + scatter (동일 trial set 평균) ──
def plot_box(ax, metric, ylabel, title):
    b_all = succ[succ['mode'] == 'Baseline'][metric].values
    c_all = succ[succ['mode'] == 'C2F'][metric].values
    b_clean = get_clean_trials('Baseline')
    c_clean = get_clean_trials('C2F')
    bp = ax.boxplot([b_all, c_all], labels=['Baseline', 'C2F'],
                    patch_artist=True, widths=0.4, showfliers=False)
    for patch, color in zip(bp['boxes'], [colors['Baseline'], colors['C2F']]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    for i, data in enumerate([b_all, c_all], 1):
        x = np.random.normal(i, 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.7, color='black', s=30, zorder=5)
    # 이상치 제거 평균 (동일 trial set)
    for i, clean in enumerate([b_clean, c_clean], 1):
        avg = clean[metric].mean()
        ax.scatter(i, avg, color='red', s=100, zorder=6, marker='D', edgecolors='white', linewidths=1)
        ax.annotate(f'avg={avg:.1f}', (i, avg), textcoords='offset points',
                    xytext=(15, 5), fontsize=9, fontweight='bold', color='red')
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ── (0,1) Time breakdown ──
plot_stacked(axes[0, 1], 'reaching_time_s', 'contact_time_s', 's', 'Avg Time Breakdown (Success)')
axes[0, 1].set_ylabel('Time (s)')

# ── (0,2) Time distribution ──
plot_box(axes[0, 2], 'task_time_s', 'Task Time (s)', 'Task Time Distribution (Success)')

# ── (1,0) Step breakdown ──
plot_stacked(axes[1, 0], 'reaching_steps', 'contact_steps', '', 'Avg Step Breakdown (Success)')
axes[1, 0].set_ylabel('Steps')

# ── (1,1) Step distribution ──
plot_box(axes[1, 1], 'task_steps', 'Task Steps', 'Task Step Distribution (Success)')

# ── (1,2) Action Hz vs Model Hz ──
ax = axes[1, 2]
has_model_hz = 'model_hz' in succ.columns
bar_w = 0.2
for i, mode in enumerate(['Baseline', 'C2F']):
    clean = get_clean_trials(mode)
    # Action Hz (= steps / wall time)
    if 'action_hz' in clean.columns:
        a_hz = clean['action_hz'].astype(float)
    else:
        a_hz = clean['reaching_steps'] / clean['reaching_time_s']
    a_mean, a_std = a_hz.mean(), a_hz.std()
    ax.bar(i - bar_w/2, a_mean, yerr=a_std, width=bar_w, color=colors[mode], capsize=4, alpha=0.7, label='Action' if i == 0 else '')
    ax.text(i - bar_w/2, a_mean + a_std + 0.3, f'{a_mean:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    # Model Hz (순수 추론 속도)
    if has_model_hz:
        m_hz = clean['model_hz'].astype(float)
        m_mean, m_std = m_hz.mean(), m_hz.std()
        ax.bar(i + bar_w/2, m_mean, yerr=m_std, width=bar_w, color=colors[mode], capsize=4, alpha=1.0, label='Model' if i == 0 else '')
        ax.text(i + bar_w/2, m_mean + m_std + 0.3, f'{m_mean:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks([0, 1]); ax.set_xticklabels(['Baseline', 'C2F'])
ax.set_ylabel('Hz'); ax.set_title('Action Hz vs Model Hz')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved: {out_path}")



# python plot_result.py \
#   /home/ailab-2204/Workspace/manipforce_c2f/eval_results/gear_insertion/baseline_16_step_delta_1step_async \
#   /home/ailab-2204/Workspace/manipforce_c2f/eval_results/gear_insertion/c2f_delta_1step_async