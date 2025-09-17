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
        stake=random.choice([1, 2, 5]),
        max_martingale=random.choice([3, 4, 5, 6]),
        sl_pips=random.choice([30, 40, 50, 60, 80]),
        tp_pips=random.choice([30, 40, 50, 60, 80]),
    )

# 读取数据
class PandasData_Minute(bt.feeds.PandasData):
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
    )

# 马丁格尔策略
class MartingaleStrategy(bt.Strategy):
    params = dict(
        stake=1,
        max_martingale=5,
        sl_pips=50,
        tp_pips=50,
        symbol='EURUSD.i',
        login=764814,
        broker_id=6,
        server_id=1,
    )

    def __init__(self):
        self.order = None
        self.martingale_level = 0
        self.last_trade_won = True
        self.trades = []
        self.ticket = 42812938  # 初始ticket，可自增

    def next(self):
        if self.order:
            return
        if not self.position:
            # 开新仓
            volume = self.params.stake * (2 ** self.martingale_level)
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
                self.close()  # backtrader 平仓
            elif self.data.close[0] >= self.tp:
                self.close_trade(reason=1)  # tp
                self.close()  # backtrader 平仓

    def close_trade(self, reason):
        close_price = self.data.close[0]
        close_time = int(self.data.datetime.datetime(0).timestamp())
        profit = (close_price - self.open_price) * (2 ** self.martingale_level) * 10000  # 简单盈亏
        commission = -0.14 * (2 ** self.martingale_level)
        swaps = -0.1 * (2 ** self.martingale_level)
        trade = [
            self.params.broker_id,
            self.ticket,
            self.params.login,
            self.params.symbol,
            self.cmd,
            (2 ** self.martingale_level),
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
        self.last_trade_won = (reason == 1)
        if self.last_trade_won:
            self.martingale_level = 0
        else:
            self.martingale_level = min(self.martingale_level + 1, self.params.max_martingale)
        self.order = None

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def stop(self):
        # 保存交易记录
        columns = ['broker_id','ticket','login','symbol','cmd','volume','open_time','open_price','sl','tp','close_time','reason','commission','swaps','close_price','profit','server_id']
        df = pd.DataFrame(self.trades, columns=columns)
        df.to_csv(output_csv, index=False)

# 加载数据
if __name__ == '__main__':
    # 确保输出文件夹存在
    output_dir = 'martingale_tradings'
    os.makedirs(output_dir, exist_ok=True)
    # 随机选品种
    symbol, data_path = get_random_symbol_and_path()
    # 读取数据
    df = pd.read_csv(
        data_path,
        parse_dates=['time'],
        date_parser=lambda x: pd.to_datetime(x, format='%Y%m%d %H%M%S')
    )
    df.set_index('time', inplace=True)
    # 随机半年区间
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
    # 随机参数
    rand_params = get_random_params()
    # 随机login、ticket
    login = random.randint(1000000000, 9999999999)
    ticket = random.randint(40000000, 49999999)
    # 输出文件名
    output_csv = os.path.join(output_dir, f"martingale_trades_{symbol}_{start_time.strftime('%Y%m%d')}_{ticket}.csv")
    # 策略参数
    strat_params = dict(
        stake=rand_params['stake'],
        max_martingale=rand_params['max_martingale'],
        sl_pips=rand_params['sl_pips'],
        tp_pips=rand_params['tp_pips'],
        symbol=symbol,
        login=login,
        broker_id=6,
        server_id=1,
    )
    cerebro = bt.Cerebro()
    data = PandasData_Minute(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(MartingaleStrategy, **strat_params)
    # 注入ticket到策略
    MartingaleStrategy.ticket = ticket
    cerebro.run()
    print(f'交易记录已保存到 {output_csv}')
