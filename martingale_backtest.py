import backtrader as bt
import pandas as pd
import os
from datetime import datetime

# 配置参数
data_path = 'history_price/eurusd.csv'  # 可更换为其他品种
output_csv = 'martingale_trades.csv'

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
    df = pd.read_csv(
        data_path,
        parse_dates=['time'],
        date_parser=lambda x: pd.to_datetime(x, format='%Y%m%d %H%M%S')
    )
    df.set_index('time', inplace=True)
    # 只取最近半年的数据
    if not df.empty:
        last_time = df.index.max()
        half_year_ago = last_time - pd.Timedelta(days=182)
        df = df[df.index >= half_year_ago]
    cerebro = bt.Cerebro()
    data = PandasData_Minute(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(MartingaleStrategy)
    cerebro.run()
    print(f'交易记录已保存到 {output_csv}')
