import backtrader as bt
import pandas as pd
import os
import random
from collections import deque
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
        p_init=random.uniform(0.45, 0.60),      # 初始胜率
        b_init=random.uniform(0.8, 1.6),        # 初始盈亏比(平均盈利/平均亏损的绝对值)
        sl_pips=random.choice([30, 40, 50, 60, 80]),
        tp_pips=random.choice([30, 40, 50, 60, 80]),
    )

def kelly_fraction(p, b, fmax=0.25):
    # f* = (b*p - (1-p)) / b = p - (1-p)/b
    if b <= 0:
        return 0.0
    f = (b * p - (1 - p)) / b
    # 常用于实盘的风控：把 Kelly 下调，如不超过 25%
    return max(0.0, min(f, fmax))

def pip_value_per_lot(symbol, price):
    """
    简化：主流直盘（EURUSD/GBPUSD/USDJPY 等），标准手每“点”价值约 $10。
    对 JPY 计价品种，点值略有不同；你可替换为更精确的计算。
    """
    if 'JPY' in symbol and price > 0:
        # 粗略：USDJPY 一点(0.01) ≈ 1000 JPY ≈ 1000/price USD
        return 1000.0 / price
    return 10.0  # USD 计价直盘近似

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
        p_init=0.52,
        b_init=1.0,
        sl_pips=50,
        tp_pips=50,
        symbol='EURUSD',
        login=764814,
        broker_id=6,
        server_id=1,
        base_balance=10000,
        ema_alpha=0.05,      # 用 EMA 平滑更新 p̂, b̂
        fmax=0.25,           # Kelly 上限（风险控制）
        max_leverage=10.0,   # 最大杠杆限制（可选）
    )

    def __init__(self):
        self.order = None
        self.trades = []
        self.ticket = 50_000_000 + random.randint(0, 9_999_999)

        # 追踪估计量（EMA）
        self.p_hat = self.params.p_init
        self.b_hat = max(1e-6, self.params.b_init)   # 避免除零
        self.alpha = self.params.ema_alpha

        # 最近成交的原始数据（可用于诊断）
        self.last_results = deque(maxlen=5000)

        # 运行时变量
        self.open_price = None
        self.open_time = None
        self.cmd = None
        self.volume = None
        self.sl = None
        self.tp = None

    def _update_edge_estimates(self, is_win, r_multiple):
        """
        is_win: bool
        r_multiple: 本次交易的 R 倍数 = 实际收益 / 预设风险(=SL pips * pip_value * volume)
        用 EMA 更新 p̂ 与 b̂（b̂ 近似=平均盈利/平均亏损绝对值）
        """
        # 胜率的 EMA
        self.p_hat = (1 - self.alpha) * self.p_hat + self.alpha * (1.0 if is_win else 0.0)

        # 用 R 倍数估盈亏比：b̂ ≈ 平均正向R / 平均负向R 的绝对值
        # 这里用简单的“分子分母的 EMA”来近似
        if not hasattr(self, '_ema_pos_r'): self._ema_pos_r = 0.0
        if not hasattr(self, '_ema_neg_r'): self._ema_neg_r = 0.0

        if r_multiple >= 0:
            self._ema_pos_r = (1 - self.alpha) * self._ema_pos_r + self.alpha * r_multiple
        else:
            self._ema_neg_r = (1 - self.alpha) * self._ema_neg_r + self.alpha * (-r_multiple)

        # 避免除零
        if self._ema_neg_r <= 1e-6:
            self.b_hat = max(1e-6, self.b_hat)  # 保持原值
        else:
            self.b_hat = max(1e-6, self._ema_pos_r / self._ema_neg_r)

    def next(self):
        if self.order or self.position:
            return

        price = float(self.data.close[0])
        equity = float(self.broker.getvalue())
        pv = pip_value_per_lot(self.params.symbol, price)

        # 计算当期 Kelly 比例（基于当前 p̂ 与 b̂）
        f_star = kelly_fraction(self.p_hat, self.b_hat, fmax=self.params.fmax)

        # 若 Kelly 为 0（或极小），则不交易（这很重要：LLM 识别到 “kelly_f_star≈0” 也应判定为 Kelly）
        if f_star <= 1e-6:
            return

        # 以 SL 风险反推手数：风险 = f* × 权益 = volume × (sl_pips × pip_value_per_lot)
        risk_per_trade = f_star * equity
        denom = max(1e-9, self.params.sl_pips * pv)
        volume = risk_per_trade / denom

        # 杠杆限制（可选）
        # 粗略杠杆 = (volume * 合约名义) / equity；这里我们仅限制 volume 防止异常
        volume = max(0.01, min(volume, 100.0))  # 防止过小/过大

        # 下单（简单做多；需要双向可自行扩展）
        self.order = self.buy(size=volume)
        self.open_price = price
        self.open_time = int(self.data.datetime.datetime(0).timestamp())
        self.cmd = 0  # buy
        self.volume = volume
        self.sl = self.open_price - self.params.sl_pips * 0.0001
        self.tp = self.open_price + self.params.tp_pips * 0.0001

    def _close_position(self, reason):
        """
        reason: 0=SL, 1=TP
        计算盈亏(含手数)、更新 Kelly 估计（用 R 倍数），并记录。
        """
        close_price = float(self.data.close[0])
        close_time = int(self.data.datetime.datetime(0).timestamp())

        price_move_pips = (close_price - self.open_price) / 0.0001  # 对 5位小数直盘
        pv = pip_value_per_lot(self.params.symbol, close_price)

        # 盈亏（USD）= pips × pip_value × volume
        profit = price_move_pips * pv * self.volume
        commission = -0.14 * max(1.0, self.volume)  # 简化：佣金与手数近似线性
        swaps = -0.1 * max(1.0, self.volume)       # 简化：隔夜费

        # 预设风险（USD）：sl_pips × pip_value × volume
        preset_risk = max(1e-6, self.params.sl_pips * pv * self.volume)
        r_multiple = profit / preset_risk

        is_win = profit > 0
        self._update_edge_estimates(is_win, r_multiple)
        self.last_results.append((is_win, r_multiple))

        # 写交易
        trade = [
            self.params.broker_id,
            self.ticket,
            self.params.login,
            self.params.symbol,
            self.cmd,
            round(self.volume, 4),
            self.open_time,
            round(self.open_price, 5),
            round(self.sl, 5),
            round(self.tp, 5),
            close_time,
            reason,
            round(commission, 2),
            round(swaps, 2),
            round(close_price, 5),
            round(profit + commission + swaps, 2),
            self.params.server_id,
            # 额外便于特征提取的字段
            round(float(self.broker.getvalue()), 2),     # equity_after
            round(self.p_hat, 6),
            round(self.b_hat, 6),
            round(kelly_fraction(self.p_hat, self.b_hat, self.params.fmax), 6)
        ]
        self.trades.append(trade)
        self.ticket += 1

        # 清理
        self.order = None
        self.open_price = None
        self.open_time = None
        self.cmd = None
        self.volume = None
        self.sl = None
        self.tp = None

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected, order.Partial]:
            self.order = None

    def next_close_logic(self):
        """分离出来便于阅读：检查 SL/TP 触发"""
        if self.position and self.open_price is not None:
            price = float(self.data.close[0])
            if price <= self.sl:
                self.close()
                self._close_position(reason=0)  # SL
            elif price >= self.tp:
                self.close()
                self._close_position(reason=1)  # TP

    def next(self):
        # 平仓逻辑
        if self.position and self.open_price is not None:
            price = float(self.data.close[0])
            if price <= self.sl:
                self.close()
                self._close_position(reason=0)  # SL
                return
            elif price >= self.tp:
                self.close()
                self._close_position(reason=1)  # TP
                return
        # 开仓逻辑
        if not self.position and not self.order:
            price = float(self.data.close[0])
            equity = float(self.broker.getvalue())
            pv = pip_value_per_lot(self.params.symbol, price)
            f_star = kelly_fraction(self.p_hat, self.b_hat, fmax=self.params.fmax)
            if f_star <= 1e-6:
                return
            risk_per_trade = f_star * equity
            denom = max(1e-9, self.params.sl_pips * pv)
            volume = risk_per_trade / denom
            volume = max(0.01, min(volume, 100.0))
            self.order = self.buy(size=volume)
            self.open_price = price
            self.open_time = int(self.data.datetime.datetime(0).timestamp())
            self.cmd = 0  # buy
            self.volume = volume
            self.sl = self.open_price - self.params.sl_pips * 0.0001
            self.tp = self.open_price + self.params.tp_pips * 0.0001

    def stop(self):
        columns = [
            'broker_id','ticket','login','symbol','cmd','volume','open_time','open_price','sl','tp',
            'close_time','reason','commission','swaps','close_price','profit','server_id',
            'equity_after','p_hat','b_hat','kelly_f_star_used'
        ]
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
    login = random.randint(1_000_000_000, 9_999_999_999)
    ticket = random.randint(50_000_000, 59_999_999)
    output_csv = os.path.join(output_dir, f"kelly_trades_{symbol}_{start_time.strftime('%Y%m%d')}_{ticket}.csv")

    strat_params = dict(
        p_init=rand_params['p_init'],
        b_init=rand_params['b_init'],
        sl_pips=rand_params['sl_pips'],
        tp_pips=rand_params['tp_pips'],
        symbol=symbol,
        login=login,
        broker_id=6,
        server_id=1,
        base_balance=10000,
        ema_alpha=0.05,
        fmax=0.25,
    )

    cerebro = bt.Cerebro()
    data = PandasData_Minute(dataname=df)
    cerebro.adddata(data)

    # 让 Broker 初始资金与 base_balance 对齐
    cerebro.broker.setcash(strat_params['base_balance'])

    cerebro.addstrategy(KellyStrategy, **strat_params)
    KellyStrategy.output_csv = output_csv
    cerebro.run()
    print(f'凯利策略交易记录已保存到 {output_csv}')
