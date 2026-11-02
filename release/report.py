"""Generate the Korean README and figures from the audited published tables."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'pairs-adf-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('SOURCE_M', 'ADF_RISK1', 'ADF_RISK2', 'ADF_RISK3')
LABELS = {'SOURCE_M': '원전략 SOURCE_M', 'ADF_RISK1': '비중복 ADF_RISK1',
          'ADF_RISK2': '최대 2개 ADF_RISK2', 'ADF_RISK3': '**최종 ADF_RISK3**'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pct(value):
    return f'{100 * float(value):.2f}%'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def row(frame, model, bps, period=None):
    selected = frame[frame.model.eq(model) & frame.cost_bps.eq(bps)]
    if period is not None:
        selected = selected[selected.period.astype(str).eq(str(period))]
    if len(selected) != 1:
        raise ValueError(f'Expected one published row: {model}/{bps}/{period}')
    return selected.iloc[0]


def make_figure(daily, yearly):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, (ax, annual) = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={'height_ratios': [1.5, 1]})
    for model, bps, label, color, style in [
        ('ADF_RISK3', 0, 'Final b=3: gross', '#126a85', '-'),
        ('ADF_RISK3', 5, 'Final b=3: 5 bp one way', '#126a85', '--'),
        ('SOURCE_M', 0, 'Original: gross', '#888888', '-'),
        ('SOURCE_M', 5, 'Original: 5 bp one way', '#888888', '--'),
    ]:
        part = daily[daily.model.eq(model) & daily.cost_bps.eq(bps)].sort_values('event_id')
        index = np.cumprod(1 + part.net_return.to_numpy())
        ax.plot(part.date, index - 1, label=label, color=color, ls=style, lw=1.6)
    ax.axhline(0, color='#555555', lw=.7)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_ylabel('Cumulative source-index return')
    ax.set_title('Full-size ADF/risk selection | Jan 2017 - May 2023', loc='left', weight='bold')
    ax.grid(axis='y', alpha=.2)
    ax.legend(ncol=2, loc='upper left', frameon=False, fontsize=9)
    years = [str(y) for y in range(2017, 2024)]
    gross = [row(yearly, 'ADF_RISK3', 0, y).cumulative_return for y in years]
    net = [row(yearly, 'ADF_RISK3', 5, y).cumulative_return for y in years]
    x = np.arange(len(years))
    annual.bar(x - .18, gross, .36, label='Final b=3: gross', color='#126a85')
    annual.bar(x + .18, net, .36, label='Final b=3: 5 bp one way', color='#db8e54')
    annual.set_xticks(x, years[:-1] + ['2023\nJan-May'])
    annual.set_ylabel('Period return')
    annual.yaxis.set_major_formatter(PercentFormatter(1))
    annual.axhline(0, color='#555555', lw=.7)
    annual.grid(axis='y', alpha=.2)
    annual.legend(ncol=2, frameon=False, fontsize=9)
    fig.text(.08, .018, 'Source-accounting index, not self-financing NAV. Original duplicate month-end rows retained.\n'
             'Fees on raw share changes; entry and terminal liquidation included. 2023 is a partial year.',
             fontsize=8, color='#555555')
    fig.tight_layout(rect=[0, .065, 1, 1])
    folder = ROOT / 'reports'
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / 'performance.png', dpi=170)
    fig.savefig(folder / 'performance.svg')
    svg = folder / 'performance.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


def main():
    costs = ROOT / 'results/costs'
    validation = json.loads((ROOT / 'checks/release_validation.json').read_text())
    if not validation['passed'] or not validation['complete'] or 'costs' not in validation:
        raise ValueError('A complete independent validation including costs is required')
    audit = json.loads((costs / 'audit.json').read_text())
    for name, expected in audit['output_sha256'].items():
        if digest(costs / name) != expected:
            raise ValueError('Cost result changed: ' + name)
    metrics = pd.read_csv(costs / 'metrics.csv')
    yearly = pd.read_csv(costs / 'yearly.csv', dtype={'period': str})
    periods = pd.read_csv(costs / 'periods.csv')
    daily = pd.read_csv(costs / 'daily.csv', parse_dates=['date'])
    make_figure(daily, yearly)
    summary = []
    for model in MODELS:
        gross, net = row(metrics, model, 0), row(metrics, model, 5)
        summary.append([LABELS[model], pct(gross.cumulative_return), pct(net.cumulative_return),
                        pct(gross.annual_vol), pct(net.annual_vol), pct(net.max_drawdown)])
    final = row(metrics, 'ADF_RISK3', 5)
    baseline = row(metrics, 'SOURCE_M', 5)
    summary_text = table(['전략', '비용 전 누적', '5bp 후 누적', '비용 전 변동성', '5bp 후 변동성', '5bp 후 최대낙폭'], summary)
    summary_text += (f'\n\n최종 모형은 같은 비용 기준의 원전략보다 누적수익이 '
                     f'**{100 * (final.cumulative_return - baseline.cumulative_return):.2f}%p** 높지만, '
                     f'기본 5bp 비용에서는 **누적 {pct(final.cumulative_return)}의 손실**이다. '
                     '비용 전 수익 개선만으로 비용 후 수익성이 확보되었다고 판단할 수 없다.')
    period_rows = []
    for key, label in [('extended_2017', '전체 2017-01-03~2023-05-31'),
                       ('2017_2019_continuation', '2017~2019'),
                       ('2020_2023_continuation', '2020~2023-05'),
                       ('2020_source_default_continuation', '원재현 기본 범위 2020-01-31~2023-05-31')]:
        source = metrics if key == 'extended_2017' else periods
        period_rows.append([label] + [pct(row(source, m, b, key).cumulative_return)
                                     for m, b in [('SOURCE_M', 0), ('SOURCE_M', 5), ('ADF_RISK3', 0), ('ADF_RISK3', 5)]])
    period_text = table(['기간', '원전략 비용 전', '원전략 5bp 후', '최종 비용 전', '최종 5bp 후'], period_rows)
    period_text += ('\n\n원재현 기본 범위 행은 원래 2020년 1월 31일 이후의 선정 구간만 취한다. '
                    '시작일의 이전 선정 행은 제외하고 새 선정 행을 포함하며, 비용은 2017년부터 이어진 포지션의 전환 비용이다. '
                    '**2020년에 현금에서 새로 시작한 백테스트가 아니다.**')
    yearly_rows = []
    for year in range(2017, 2024):
        yearly_rows.append([str(year) if year < 2023 else '2023 (1~5월)'] +
                           [pct(row(yearly, m, b, year).cumulative_return)
                            for m, b in [('SOURCE_M', 0), ('SOURCE_M', 5), ('ADF_RISK3', 0), ('ADF_RISK3', 5)]])
    sensitivity = table(['편도 비용', '원전략', 'ADF_RISK1', 'ADF_RISK2', '최종 ADF_RISK3'],
                        [[f'{bps}bp ({bps / 100:.2f}%)'] +
                         [pct(row(metrics, model, bps).cumulative_return) for model in MODELS]
                         for bps in (0, 1, 5, 10)])
    stale_ratio = final.stale_notional_sum / (final.order_notional_sum + final.terminal_notional_sum)
    stale_dates = ', '.join(audit['stale_events_by_model']['ADF_RISK3'])
    cost_text = (f'최종 모형에서 직전 관측 가격을 사용한 주문 날짜는 **{stale_dates}**이며, '
                 f'전체 주문대금 중 해당 평가 금액 비중은 **{stale_ratio * 100:.5f}%**다. '
                 '원가격은 변경하지 않았다. 수수료를 차감해도 이후 신호·선정·원주식 수·원래 자본은 바뀌지 않는다. '
                 '즉, 수수료 차감으로 줄어든 NAV에 맞춰 미래 포지션을 재조정하는 회계는 적용하지 않았다.')
    validation_text = (
        f'- **77개월 × 4개 모형**의 선정 개수·중복 상한·점수·학습 위험 제약과 원거래 재계산을 확인했다.\n'
        f'- 원소스·입력·확정 결과 **{validation["protected_files"]}개 파일**의 해시를 확인했다.\n'
        f'- 비용률별 행 **{validation["costs"]["rows"]:,}개**와 기간 집계 '
        f'**{validation["costs"]["checked_summary_rows"]:,}개**의 비용식·누적수익·변동성·샤프·낙폭을 대조했다.\n'
        '- 비용 0에서 기존 4개 모형의 비용 전 성과와 일치한다.\n'
        '- 검산 기록: [release_validation.json](checks/release_validation.json), '
        '[비용 계산 audit.json](results/costs/audit.json).\n'
        '- 최초 게시 시 13개 단위 테스트와 작은 그래프 36개 전수탐색 대조를 통과했다. '
        '별도 OLS·Q 계산으로 실제 주문 20개 사건도 대조했다. '
        '[테스트 기록](checks/test_validation.json), '
        '[최적화기 검산](checks/optimizer_validation.json), '
        '[첫 달 후보 재생성·재최적화 기록](checks/new_optimization_smoke.json).\n'
        '- 작은 사례의 비용·실행 검산과 최적화기 전수탐색 대조는 위 명령으로 별도 실행할 수 있다.'
    )
    replacements = {'SUMMARY': summary_text, 'PERIODS': period_text,
                    'YEARLY': table(['연도', '원전략 비용 전', '원전략 5bp 후', '최종 비용 전', '최종 5bp 후'], yearly_rows),
                    'SENSITIVITY': sensitivity, 'COST_AUDIT': cost_text, 'VALIDATION': validation_text}
    text = (ROOT / 'release/README.template.md').read_text()
    for marker, content in replacements.items():
        token = '<!-- ' + marker + ' -->'
        if text.count(token) != 1:
            raise ValueError('README template marker must be unique: ' + marker)
        text = text.replace(token, content)
    if '<!--' in text:
        raise ValueError('Unresolved template marker')
    (ROOT / 'README.md').write_text(text)
    for target in re.findall(r'\]\(([^)]+)\)', text):
        if not target.startswith(('http://', 'https://', '#')) and not (ROOT / target).exists():
            raise ValueError('Broken local README link: ' + target)
    sources = ['results/costs/' + name for name in ('metrics.csv', 'yearly.csv', 'monthly.csv', 'periods.csv', 'daily.csv', 'audit.json')]
    sources += ['checks/release_validation.json', 'release/README.template.md', 'release/report.py']
    (ROOT / 'reports/report_manifest.json').write_text(json.dumps({
        'source_sha256': {name: digest(ROOT / name) for name in sources},
        'generated_sha256': {name: digest(ROOT / name) for name in
                             ['README.md', 'reports/performance.png', 'reports/performance.svg']},
        'interpretation': 'Source-accounting gross and after-cost indices, not self-financing NAV'
    }, indent=2) + '\n')
    print(json.dumps({'generated': ['README.md', 'reports/performance.png', 'reports/performance.svg'],
                      'final_gross': float(row(metrics, 'ADF_RISK3', 0).cumulative_return),
                      'final_net_5bp': float(final.cumulative_return)}))


if __name__ == '__main__':
    main()
