import argparse
import json
import logging
from datetime import date

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def date_range_to_years(startdate, enddate):
    """Make sure input is iso format e.g. slice the tuned datetime '2017-08-01T00:00:00Z'[:10]"""
    s = date.fromisoformat(startdate)
    e = date.fromisoformat(enddate)
    td = e - s
    return td.days / 365


def cagr(start, end, period):
    """E.g. start 1 end 73.62 period 3 years equals 319.11% cagr"""
    return ((end / start) ** (1 / period) - 1) * 100


sns.set(style="darkgrid")
plt.style.use("dark_background")
plt.rcParams.update(
    {"grid.linewidth": 0.5, "grid.alpha": 0.5, "font.family": "monospace", "font.size": 10,
     'axes.titlesize': 'medium', 'axes.labelsize': 'medium', 'xtick.labelsize': 'small', 'ytick.labelsize': 'small'})

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', default='data', type=str,
                    help='Dir name where data is located e.g. data')
parser.add_argument('--plotdir', default='plots', type=str,
                    help='Dir name where plots are output to e.g. plots')
parser.add_argument('--file', default='tuned.json', type=str,
                    help='Filename of Tuned json data, default tuned.json in data/')
parser.add_argument('--oos', type=str, help='Plot grey line for out of sample period start date e.g. 20220101')
parser.add_argument('--forward', type=str, help='Plot blue line for forward test period start date e.g. 20220201')
parser.add_argument('--live', type=str, help='Plot green line for live period start date e.g. 20220301')
parser.add_argument('--resample', default='D', choices=['D', 'W', 'M'], type=str,
                    help='Resample equity curve using pandas period W, M etc')
args = parser.parse_args()

with open(f'{args.datadir}/{args.file}') as f:
    bt_data = json.load(f)

data = []
for t in bt_data['trades']:
    d = dict()
    try:
        d['Date'] = t['orders'][1]['placedTime']
        d['Percent'] = t['profitPercentage']
        d['Balance'] = t['accumulatedBalance']
        d['Pnl'] = t['compoundProfitPerc']
    except (IndexError, KeyError):
        logger.debug("Trades data has an open trade")
        continue
    d['Side'] = 1 if t['orders'][0]['side'] in ['Buy', 'Long'] else -1
    d['SL'] = 1 if 'source' in t['orders'][1] and 'STOPLOSS' in t['orders'][1]['source'] else 0
    d['TP'] = 1 if 'source' in t['orders'][1] and 'TAKE_PROFIT' in t['orders'][1]['source'] else 0
    data.append(d)

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Resample to daily as there may be multiple trades in a day and this adds flat days so chart looks ok
daily = df.resample('D').last().fillna(method='pad')
daily['Pnl'] = daily['Pnl'] * 100

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 6),
                                             gridspec_kw={'height_ratios': [2, 1], 'width_ratios': [3, 1]})

sns.lineplot(data=daily.resample(args.resample).last()['Pnl'], ax=ax1, color='b')


def period_line(date, text, colour):
    ax1.axvline(x=pd.to_datetime(date), linewidth=0.75, linestyle='--', color=colour)
    ax1.text(pd.to_datetime(date), 0, text, horizontalalignment='left',
             verticalalignment='center', fontsize=8, color=colour, style='italic')


if args.oos:
    period_line(args.oos, f' Out of sample\n period start', 'lightgrey')

if args.forward:
    period_line(args.forward, f' Forward test\n period start', 'lightblue')

if args.live:
    period_line(args.live, f' Live\n period start', 'green')

start_date = daily.index[0].date()
days = len(daily)
profit = bt_data['performance']['profitPerc'] * 100
dd = bt_data['performance']['maxDrawdown'] * 100
win_rate = bt_data['performance']['profitableTrades'] * 100
profitable_months = bt_data['performance']['profitableMonths'] * 100
avg_monthly_profit = bt_data['performance']['avgMonthlyProfit'] * 100
avg_win_month = bt_data['performance']['avgWinMonth'] * 100
try:
    avg_lose_month = bt_data['performance']['avgLoseMonth'] * 100
except TypeError:
    avg_lose_month = 0
avg_trade = df['Percent'].mean() * 100
avg_win = df[df['Percent'] > 0]['Percent'].mean() * 100
avg_loss = df[df['Percent'] < 0]['Percent'].mean() * 100
min_trade = df['Percent'].min() * 100
max_trade = df['Percent'].max() * 100
sl = df['SL'].sum()
tp = df['TP'].sum()

years = date_range_to_years(bt_data['trades'][0]['orders'][0]['placedTime'][:10],
                            bt_data['trades'][-1]['orders'][0]['placedTime'][:10])
cagr = cagr(bt_data['performance']['startAllocation'], bt_data['performance']['endAllocation'], years)
sharpe = daily['Percent'].mean() / daily['Percent'].std() * np.sqrt(365)
sortino = daily['Percent'].mean() / daily[daily < 0]['Percent'].std() * np.sqrt(365)

text = f"{args.file}\n" \
       f"Start date: {start_date} ({days} days)\n" \
       f"Profit: {profit:.1f}%\n" \
       f"DD: {dd:.1f}%\n" \
       f"Win rate: {win_rate:.2f}%\n" \
       f"Profitable months: {profitable_months:.1f}%\n" \
       f"Avg monthly profit: {avg_monthly_profit:.2f}%\n" \
       f"Avg win month: {avg_win_month:.2f}%\n" \
       f"Avg lose month: {avg_lose_month:.2f}%\n" \
       f"Avg trade/win/loss: {avg_trade:.2f}%/{avg_win:.2f}%/{avg_loss:.2f}%\n" \
       f"Min/max trade: {min_trade:.2f}%/{max_trade:.2f}%\n" \
       f"SL/TP: {sl}/{tp}\n" \
       f"CAGR: {cagr:.2f}\n" \
    # f"Sharpe: {sharpe:.2f}\n" \
# f"Sortino: {sortino:.2f}\n"
ax1.text(0.02, 0.95, text, horizontalalignment='left', verticalalignment='top', transform=ax1.transAxes)

ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.text(0.5, 0.5, f'@1337Research', fontsize=10, color='grey', horizontalalignment='center',
         transform=ax2.transAxes, fontfamily='monospace', fontweight='bold', fontstyle='italic')

monthly = df[['Balance']].resample('M').last()
# Have to hack the monthly and add initial alloc as start of month so pct_change() will show the first change not na
starting_amount = bt_data['performance']['startAllocation']
monthly.loc[monthly.index[0] - pd.offsets.MonthBegin()] = [starting_amount]
monthly.sort_index(inplace=True)

monthly_change = monthly.pct_change().dropna().round(4) * 100

sns.barplot(x=monthly_change.index, y='Balance', data=monthly_change, ax=ax3, color='b',
            edgecolor='b')  # have to set blue else will use rainbow

# Add amounts to tops of bars
ax3.bar_label(ax3.containers[0], fontsize='small')  # explicitly set fontsize here, can't find settings param
ax3.set_xticklabels(labels=monthly_change.index.strftime("'%y%m"), ha='center')
ax3.set_ylabel("Pnl")
ax3.set_title("Monthly returns")

for ax in [ax1, ax3]:
    ax.margins(x=0.02)  # Sync the margin otherwise lineplot has more than barplot
    ax.set(xlabel=None)  # Hide x Date labels, more minimal

# Need to specify alpha as default has some transparency. binwidth in 1% steps and use shrink to add gaps
sns.histplot(df['Percent'], ax=ax4, edgecolor='b', alpha=1, binwidth=0.01, shrink=0.6)

ax4.set_title("Distribution of returns")
ax4.set(xlabel=None)

sns.despine()

plt.tight_layout()
save_filename = f'{args.plotdir}/{args.file}.jpg'
plt.savefig(save_filename, dpi=200)
logger.info(f'Saved {save_filename}')
plt.show()
