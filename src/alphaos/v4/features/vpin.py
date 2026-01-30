"""
VPIN (Volume-Synchronized Probability of Informed Trading)

基于 BVC (Bulk Volume Classification) 方法计算订单流毒性指标。

VPIN 原理：
1. 将数据按固定成交量划分为 Volume Buckets
2. 使用价格变动分布估算买卖量
3. 计算订单失衡 OI = |V_buy - V_sell|
4. VPIN = mean(OI) / V over n buckets

参考：
- 降噪LNN特征提取与信号过滤.md Section 4.1
- Easley et al. (2012) "Flow Toxicity and Liquidity"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from alphaos.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VPINConfig:
    """
    VPIN 计算配置
    
    Args:
        bucket_volume: 单个 bucket 的目标成交量
        n_buckets: 用于 VPIN 计算的 bucket 数量
        sigma_window: 用于估算 dp 标准差的窗口
        use_tick_count: 是否使用 tick 计数代替真实成交量
    """
    bucket_volume: float = 1000.0
    n_buckets: int = 50
    sigma_window: int = 50
    use_tick_count: bool = True  # Sim2Real: MT5 volume=0 回退
    
    def to_dict(self) -> dict:
        return {
            "bucket_volume": self.bucket_volume,
            "n_buckets": self.n_buckets,
            "sigma_window": self.sigma_window,
            "use_tick_count": self.use_tick_count,
        }


@dataclass
class VolumeBucket:
    """
    Volume Bucket 数据结构
    """
    start_idx: int = 0
    end_idx: int = 0
    volume: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    price_change: float = 0.0
    sigma: float = 1e-6
    
    @property
    def order_imbalance(self) -> float:
        """订单失衡 |V_buy - V_sell|"""
        return abs(self.buy_volume - self.sell_volume)
    
    @property
    def oi_ratio(self) -> float:
        """标准化订单失衡"""
        if self.volume > 0:
            return self.order_imbalance / self.volume
        return 0.0


@dataclass
class VPINCalculator:
    """
    VPIN 计算器
    
    支持两种模式：
    1. 批量计算（训练）：一次性处理所有数据
    2. 流式计算（推理）：逐 bar 更新
    
    使用方式（批量）：
    ```python
    calc = VPINCalculator(VPINConfig())
    vpin_series = calc.compute_batch(closes, volumes)
    ```
    
    使用方式（流式）：
    ```python
    calc = VPINCalculator(VPINConfig())
    for close, volume in data:
        vpin = calc.update(close, volume)
    ```
    """
    config: VPINConfig = field(default_factory=VPINConfig)
    
    # 内部状态
    _buckets: list[VolumeBucket] = field(default_factory=list, init=False)
    _current_bucket: VolumeBucket | None = field(default=None, init=False)
    _cumulative_volume: float = field(default=0.0, init=False)
    _price_buffer: list[float] = field(default_factory=list, init=False)
    _last_price: float = field(default=0.0, init=False)
    _bar_idx: int = field(default=0, init=False)
    
    def compute_batch(
        self,
        closes: NDArray[np.float64],
        volumes: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """
        批量计算 VPIN 序列
        
        Args:
            closes: 收盘价序列
            volumes: 成交量序列（可选，默认使用 tick count）
            
        Returns:
            VPIN 值序列（与 closes 对齐）
        """
        n = len(closes)
        
        # 处理成交量
        if volumes is None or self.config.use_tick_count:
            # 每个 bar 视为 1 单位成交量（退化为 tick count）
            volumes = np.ones(n, dtype=np.float64)
        
        # 计算价格变动
        log_returns = np.zeros(n, dtype=np.float64)
        log_returns[1:] = np.log(closes[1:] / closes[:-1])
        
        # 滚动标准差（用于 BVC）
        rolling_sigma = self._compute_rolling_sigma(log_returns)
        
        # 构建 Volume Buckets
        buckets = self._build_buckets_batch(
            closes, volumes, log_returns, rolling_sigma
        )
        
        # 计算 VPIN 序列
        vpin = self._compute_vpin_from_buckets(buckets, n, volumes)
        
        return vpin
    
    def _compute_rolling_sigma(
        self, 
        log_returns: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """计算滚动标准差"""
        n = len(log_returns)
        window = self.config.sigma_window
        
        sigma = np.zeros(n, dtype=np.float64)
        
        # 初始阶段使用扩展窗口
        for i in range(n):
            start = max(0, i - window + 1)
            if i - start > 1:
                sigma[i] = np.std(log_returns[start:i+1])
            else:
                sigma[i] = 1e-6
        
        # 防止除零
        sigma = np.maximum(sigma, 1e-6)
        
        return sigma
    
    def _build_buckets_batch(
        self,
        closes: NDArray[np.float64],
        volumes: NDArray[np.float64],
        log_returns: NDArray[np.float64],
        sigmas: NDArray[np.float64],
    ) -> list[VolumeBucket]:
        """批量构建 Volume Buckets"""
        buckets = []
        target = self.config.bucket_volume
        
        current = VolumeBucket()
        cum_volume = 0.0
        start_idx = 0
        
        for i in range(len(closes)):
            vol = volumes[i]
            dp = log_returns[i]
            sigma = sigmas[i]
            
            # BVC: 使用正态分布 CDF 估算买卖量
            # V_buy = V * Phi(dp / sigma)
            # V_sell = V * (1 - Phi(dp / sigma))
            if sigma > 0:
                z = dp / sigma
                prob_buy = stats.norm.cdf(z)
            else:
                prob_buy = 0.5
            
            v_buy = vol * prob_buy
            v_sell = vol * (1 - prob_buy)
            
            # 累积到当前 bucket
            cum_volume += vol
            current.buy_volume += v_buy
            current.sell_volume += v_sell
            current.price_change += dp
            current.sigma = sigma
            current.end_idx = i
            
            # 检查 bucket 是否满
            if cum_volume >= target:
                current.start_idx = start_idx
                current.volume = cum_volume
                buckets.append(current)
                
                # 开始新 bucket
                current = VolumeBucket()
                cum_volume = 0.0
                start_idx = i + 1
        
        # 处理最后一个不完整的 bucket
        if cum_volume > 0:
            current.start_idx = start_idx
            current.volume = cum_volume
            buckets.append(current)
        
        return buckets
    
    def _compute_vpin_from_buckets(
        self,
        buckets: list[VolumeBucket],
        n_samples: int,
        volumes: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """从 buckets 计算 VPIN 序列"""
        vpin = np.zeros(n_samples, dtype=np.float64)
        n_buckets = self.config.n_buckets
        
        if len(buckets) < n_buckets:
            # 不足 n_buckets，返回全零
            return vpin
        
        # 为每个位置计算 VPIN
        bucket_idx = n_buckets - 1
        
        for b_idx in range(n_buckets - 1, len(buckets)):
            bucket = buckets[b_idx]
            
            # 计算过去 n_buckets 的 VPIN
            oi_sum = 0.0
            vol_sum = 0.0
            for j in range(b_idx - n_buckets + 1, b_idx + 1):
                oi_sum += buckets[j].order_imbalance
                vol_sum += buckets[j].volume
            
            if vol_sum > 0:
                current_vpin = oi_sum / vol_sum
            else:
                current_vpin = 0.0
            
            # 填充对应的样本索引
            start = bucket.start_idx
            end = bucket.end_idx + 1
            vpin[start:end] = current_vpin
        
        # 前向填充初始值
        first_valid = 0
        for i in range(n_samples):
            if vpin[i] > 0:
                first_valid = i
                break
        
        if first_valid > 0:
            vpin[:first_valid] = vpin[first_valid]
        
        return vpin
    
    def update(self, close: float, volume: float = 1.0) -> float:
        """
        流式更新 VPIN（单 bar）
        
        Args:
            close: 当前收盘价
            volume: 当前成交量（默认 1.0 = tick count）
            
        Returns:
            当前 VPIN 值
        """
        self._bar_idx += 1
        
        # 使用 tick count 如果配置
        if self.config.use_tick_count:
            volume = 1.0
        
        # 计算价格变动
        if self._last_price > 0:
            dp = math.log(close / self._last_price)
        else:
            dp = 0.0
        
        self._last_price = close
        
        # 更新价格缓冲（用于 sigma 估算）
        self._price_buffer.append(dp)
        if len(self._price_buffer) > self.config.sigma_window:
            self._price_buffer.pop(0)
        
        # 估算 sigma
        if len(self._price_buffer) > 1:
            sigma = np.std(self._price_buffer)
        else:
            sigma = 1e-6
        sigma = max(sigma, 1e-6)
        
        # BVC 估算
        if sigma > 0:
            z = dp / sigma
            prob_buy = stats.norm.cdf(z)
        else:
            prob_buy = 0.5
        
        v_buy = volume * prob_buy
        v_sell = volume * (1 - prob_buy)
        
        # 初始化当前 bucket
        if self._current_bucket is None:
            self._current_bucket = VolumeBucket(start_idx=self._bar_idx)
        
        # 累积到当前 bucket
        self._cumulative_volume += volume
        self._current_bucket.buy_volume += v_buy
        self._current_bucket.sell_volume += v_sell
        self._current_bucket.price_change += dp
        self._current_bucket.sigma = sigma
        self._current_bucket.end_idx = self._bar_idx
        
        # 检查 bucket 是否满
        if self._cumulative_volume >= self.config.bucket_volume:
            self._current_bucket.volume = self._cumulative_volume
            self._buckets.append(self._current_bucket)
            
            # 保持 bucket 缓冲区大小
            if len(self._buckets) > self.config.n_buckets * 2:
                self._buckets = self._buckets[-self.config.n_buckets:]
            
            # 开始新 bucket
            self._current_bucket = VolumeBucket(start_idx=self._bar_idx + 1)
            self._cumulative_volume = 0.0
        
        # 计算当前 VPIN
        return self._compute_current_vpin()
    
    def _compute_current_vpin(self) -> float:
        """计算当前 VPIN 值"""
        n_buckets = self.config.n_buckets
        
        if len(self._buckets) < n_buckets:
            return 0.0
        
        # 使用最近 n_buckets 个完整 bucket
        recent_buckets = self._buckets[-n_buckets:]
        
        oi_sum = sum(b.order_imbalance for b in recent_buckets)
        vol_sum = sum(b.volume for b in recent_buckets)
        
        if vol_sum > 0:
            return oi_sum / vol_sum
        return 0.0
    
    @property
    def current_vpin(self) -> float:
        """获取当前 VPIN 值"""
        return self._compute_current_vpin()
    
    @property
    def bucket_count(self) -> int:
        """已完成的 bucket 数量"""
        return len(self._buckets)
    
    def reset(self) -> None:
        """重置计算器"""
        self._buckets.clear()
        self._current_bucket = None
        self._cumulative_volume = 0.0
        self._price_buffer.clear()
        self._last_price = 0.0
        self._bar_idx = 0
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "bucket_count": len(self._buckets),
            "current_vpin": round(self.current_vpin, 4),
            "cumulative_volume": round(self._cumulative_volume, 2),
            "bar_idx": self._bar_idx,
            "sigma_buffer_size": len(self._price_buffer),
        }
