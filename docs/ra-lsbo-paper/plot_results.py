"""Plot the released RA-LSBO paper data. Requires matplotlib."""

import argparse
import csv
import gzip
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parent
SCENARIOS = {'headwind': 'Headwind', 'payload': 'Increased payload'}
ARMS = {
    'ra_structured_greedy': ('RA-LSBO', '#087f8c', '-'),
    'archsbo_screened': ('Screened ArchSBO', '#a54820', '--'),
    'logcei_screened_online': ('Adapted LogCEI', '#66518d', ':'),
    'lsbo_native': ('Original LSBO', '#a54820', '--'),
    'ra_refine_interval': ('Interval-only RA-LSBO', '#66518d', ':'),
    'ra_structured_sign_greedy': ('Structured sign-only RA-LSBO', '#a54820', '--'),
}
EXTERNAL = ('ra_structured_greedy', 'archsbo_screened', 'logcei_screened_online')
INTERNAL = ('ra_structured_greedy', 'lsbo_native', 'ra_refine_interval')


def load(name):
    path = ROOT / 'data' / name
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def matching(rows, **filters):
    return [row for row in rows if all(row[key] == value for key, value in filters.items())]


def points(rows):
    return sorted(rows, key=lambda row: int(row['evaluation']))


def style_axis(ax):
    ax.grid(axis='y', color='#d7e0e5', linewidth=.75)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)


def plot_attainment(ax, data, profile, scenario, cohort, arms, budget):
    series = matching(data, experiment=profile, scenario=scenario, cohort=cohort)
    for arm in arms:
        title, color, style = ARMS[arm]
        rows = points(matching(series, arm=arm))
        ax.step([int(r['evaluation']) for r in rows], [float(r['target_fraction']) for r in rows],
                where='post', label=title, color=color, linestyle=style, lw=2.1)
    cutoff = float(series[0]['threshold_kWh'])
    n = int(series[0]['seeds'])
    ax.set_title(f'{SCENARIOS[scenario]}  |  ≤{cutoff:g} kWh  |  n={n}', loc='left', fontsize=10)
    ax.set(xlim=(8, budget), ylim=(0, 1.04), xticks=list(range(8, budget + 1, 8)))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    style_axis(ax)


def error_series(ax, rows, label, color, style, scale=1., shift=0.):
    rows = points([row for row in rows if row['mean_delta']])
    x = [int(r['evaluation']) + shift for r in rows]
    y = [float(r['mean_delta']) / scale for r in rows]
    lower = [v - float(r['ci95_low']) / scale for v, r in zip(y, rows)]
    upper = [float(r['ci95_high']) / scale - v for v, r in zip(y, rows)]
    ax.errorbar(x, y, yerr=[lower, upper], color=color, label=label,
                linestyle=style, marker='o', ms=4, capsize=3, linewidth=1.55)


def save(fig, output, name):
    fig.savefig(output / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(output / f'{name}.png', dpi=170, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'figures')
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.labelcolor': '#263746', 'text.color': '#263746',
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    attained = load('target_attainment.csv.gz')
    energy = load('paired_energy_checkpoints.csv')
    ablation = load('mechanism_ablation.csv')

    fig, axes = plt.subplots(2, 2, figsize=(9.3, 6.2), layout='constrained')
    fig.suptitle('Energy-target attainment in the 32-evaluation cold-start runs', fontsize=14)
    for col, scenario in enumerate(SCENARIOS):
        plot_attainment(axes[0, col], attained, 'internal_32', scenario, 'cold', INTERNAL, 32)
        plot_attainment(axes[1, col], attained, 'external_32', scenario, 'cold', EXTERNAL, 32)
        axes[1, col].set_xlabel('Cumulative design evaluations')
    for row, legend in zip(axes, ['Internal comparison', 'External comparison']):
        row[0].set_ylabel('Seeds reaching the energy target')
        row[0].legend(loc='upper left', title=legend, fontsize=7, title_fontsize=8)
    save(fig, output, '01_cold_start_target_attainment')

    fig, axes = plt.subplots(2, 2, figsize=(9.3, 6.25), layout='constrained')
    fig.suptitle('Budget sensitivity in the 64-evaluation rerun', fontsize=14)
    for col, scenario in enumerate(SCENARIOS):
        plot_attainment(axes[0, col], attained, 'external_64', scenario, 'all', EXTERNAL, 64)
        axes[0, col].set_xlabel('Cumulative design evaluations')
        ax = axes[1, col]
        for offset, control in zip([-.6, .6], EXTERNAL[1:]):
            rows = matching(energy, experiment='external_64', cohort='all', scenario=scenario, comparator=control)
            rows = [row for row in rows if int(row['evaluation']) in (32, 48, 64)]
            title, color, style = ARMS[control]
            error_series(ax, rows, title, color, style, scale=1000., shift=offset)
        ax.axhline(0, color='#617583', lw=.9)
        ax.set(xlim=(29, 67), xticks=(32, 48, 64), xlabel='Cumulative design evaluations')
        ax.set_title(f'{SCENARIOS[scenario]}  |  paired best-energy difference', loc='left', fontsize=10)
        style_axis(ax)
    axes[0, 0].set_ylabel('Seeds reaching the energy target')
    axes[0, 0].legend(loc='lower right', fontsize=8)
    axes[1, 0].set_ylabel('RA-LSBO minus comparator (kWh)')
    axes[1, 0].legend(loc='lower right', fontsize=8)
    axes[1, 0].text(.20, .92, '95% paired bootstrap intervals', transform=axes[1, 0].transAxes,
                    fontsize=7, va='top', bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1})
    save(fig, output, '02_external_budget_sensitivity')

    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.65), layout='constrained', sharey=True)
    fig.suptitle('Completion-risk prediction on held-out designs', fontsize=14)
    for ax, scenario in zip(axes, SCENARIOS):
        for offset, control in zip([-.10, .10], ('ra_refine_interval', 'ra_structured_sign_greedy')):
            title, color, style = ARMS[control]
            series = matching(ablation, scenario=scenario, metric='balanced_brier', comparator=control)
            error_series(ax, series, title, color, style, shift=offset)
        n = matching(ablation, scenario=scenario, metric='balanced_brier')[0]['pairs']
        ax.set_title(f'{SCENARIOS[scenario]}  |  n={n} cold starts', loc='left', fontsize=10)
        ax.axhline(0, color='#617583', lw=.9)
        ax.set(xticks=(8, 12, 16), xlabel='Cumulative design evaluations')
        style_axis(ax)
    axes[0].set_ylabel('RA-LSBO minus comparator (Brier score)')
    axes[1].legend(loc='lower left', fontsize=7)
    save(fig, output, '03_completion_risk_ablation')


if __name__ == '__main__':
    main()
