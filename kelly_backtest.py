import backtrader as bt
import pandas as pd
import os
import random
from datetime import datetime, timedelta

# 随机选取品种和参数
def get_random_symbol_and_path():
    folder = 'history_price'
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    symbol_file = random.choice(files)
    symbol = symbol_file.replace('.csv', '').upper()
    return symbol, os.path.join(folder, symbol_file)

def get_random_params():
    return dict(
        win_rate=random.uniform(0.4, 0.7),  # 假设胜率40%-70%
        win_loss_ratio=random.uniform(0.8, 2.0),  # 盈亏比0.8-2.0
        sl_pips=random.choice([30, 40, 50, 60, 80]),
        tp_pips=random.choice([30, 40, 50, 60, 80]),
    )

def kelly_fraction(win_rate, win_loss_ratio):
    # 凯利公式: f* = (bp - q)/b
    # b=盈亏比, p=胜率, q=1-p
    b = win_loss_ratio
    p = win_rate
    q = 1 - p
    kf = (b * p - q) / b if b > 0 else 0
    return max(0, min(kf, 1))  # 限制在[0,1]

class PandasData_Minute(bt.feeds.PandasData):
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
    )

class KellyStrategy(bt.Strategy):
    params = dict(
        win_rate=0.5,
        win_loss_ratio=1.0,
        sl_pips=50,
        tp_pips=50,
        symbol='EURUSD.i',
        login=764814,
        broker_id=6,
        server_id=1,
        base_balance=10000,
    )

    def __init__(self):
        self.order = None
        self.trades = []
        self.ticket = 50000000 + random.randint(0, 9999999)
        self.balance = self.params.base_balance

    def next(self):
        if self.order:
            return
        if not self.position:
            # 用凯利公式决定本次投入比例
            kf = kelly_fraction(self.params.win_rate, self.params.win_loss_ratio)
            volume = max(1, int(self.balance * kf * 0.01))  # 假设1手=1%资金
            self.order = self.buy(size=volume)
            self.open_price = self.data.close[0]
            self.open_time = int(self.data.datetime.datetime(0).timestamp())
            self.sl = self.open_price - self.params.sl_pips * 0.0001
            self.tp = self.open_price + self.params.tp_pips * 0.0001
            self.cmd = 0  # buy
        else:
            # 检查止盈止损
            if self.data.close[0] <= self.sl:
                self.close_trade(reason=0)  # sl
                self.close()
            elif self.data.close[0] >= self.tp:
                self.close_trade(reason=1)  # tp
                self.close()

    def close_trade(self, reason):
        close_price = self.data.close[0]
        close_time = int(self.data.datetime.datetime(0).timestamp())
        profit = (close_price - self.open_price) * 10000  # 简单盈亏
        commission = -0.14
        swaps = -0.1
        trade = [
            self.params.broker_id,
            self.ticket,
            self.params.login,
            self.params.symbol,
            self.cmd,
            1,
            self.open_time,
            round(self.open_price, 5),
            round(self.sl, 5),
            round(self.tp, 5),
            close_time,
            reason,
            commission,
            swaps,
            round(close_price, 5),
            round(profit, 2),
            self.params.server_id
        ]
        self.trades.append(trade)
        self.ticket += 1
        self.balance += profit
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def stop(self):
        columns = ['broker_id','ticket','login','symbol','cmd','volume','open_time','open_price','sl','tp','close_time','reason','commission','swaps','close_price','profit','server_id']
        df = pd.DataFrame(self.trades, columns=columns)
        df.to_csv(self.output_csv, index=False)

if __name__ == '__main__':
    output_dir = 'kelly_tradings'
    os.makedirs(output_dir, exist_ok=True)
    symbol, data_path = get_random_symbol_and_path()
    df = pd.read_csv(
        data_path,
        parse_dates=['time'],
        date_parser=lambda x: pd.to_datetime(x, format='%Y%m%d %H%M%S')
    )
    df.set_index('time', inplace=True)
    if not df.empty:
        min_time = df.index.min()
        max_time = df.index.max() - pd.Timedelta(days=182)
        if min_time >= max_time:
            start_time = min_time
        else:
            start_time = min_time + (max_time - min_time) * random.random()
        start_time = pd.to_datetime(start_time)
        end_time = start_time + pd.Timedelta(days=182)
        df = df[(df.index >= start_time) & (df.index < end_time)]
    rand_params = get_random_params()
    login = random.randint(1000000000, 9999999999)
    ticket = random.randint(50000000, 59999999)
    output_csv = os.path.join(output_dir, f"kelly_trades_{symbol}_{start_time.strftime('%Y%m%d')}_{ticket}.csv")
    strat_params = dict(
        win_rate=rand_params['win_rate'],
        win_loss_ratio=rand_params['win_loss_ratio'],
        sl_pips=rand_params['sl_pips'],
        tp_pips=rand_params['tp_pips'],
        symbol=symbol,
        login=login,
        broker_id=6,
        server_id=1,
        base_balance=10000,
    )
    cerebro = bt.Cerebro()
    data = PandasData_Minute(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(KellyStrategy, **strat_params)
    # 注入输出文件名
    KellyStrategy.output_csv = output_csv
    cerebro.run()
    print(f'凯利策略交易记录已保存到 {output_csv}')
