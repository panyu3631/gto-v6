"""
GTO-GameFlow v5.0 滚动窗口 LASSO 因子选择模块

基于滚动窗口(rolling window)的 L1 正则化(LASSO)逻辑回归,
自动筛选每个联赛在每个时间窗口内的有效因子,
剔除冗余或噪声因子,输出稀疏的因子权重字典。

核心概念
--------
滚动窗口 (Rolling Window):
  在时间序列数据中,使用固定长度的历史窗口数据训练模型,
  窗口随时间向前滚动。例如窗口大小 = 90 天,则每次训练仅使用
  最近 90 天的比赛数据,确保模型参数反映最新的市场状态。

无前瞻偏差 (No Look-Ahead Bias):
  训练窗口严格以比赛日期为界,仅使用该场比赛之前的历史数据。
  绝不使用未来数据来预测过去的结果,这是体育博彩模型化的
  核心约束。违反此约束会导致回测结果虚高,实战中失效。

LASSO 因子选择:
  L1 正则化将不相关或冗余因子的系数压缩至零,实现自动因子
  筛选。选中的因子(非零系数)权重设为 1.0,被剔除的因子
  权重设为 0.0,形成二元权重字典。

用法示例
--------
    >>> from src.factors.lasso_selector import (
    ...     RollingLassoSelector, RollingWindowCache, build_training_data_from_matches
    ... )
    >>>
    >>> selector = RollingLassoSelector(window_size=90)
    >>> X, y, factor_names = build_training_data_from_matches(matches)
    >>> weights = selector.select("premier_league", X, y, factor_names, alpha=0.01)
    >>> print(weights)  # {"F1": 1.0, "F3": 1.0, "F9": 1.0, "F2": 0.0, ...}
"""

import logging
import warnings
import sys
import os
from typing import Dict, List, Optional, Tuple, Union

# 确保 src 可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.utils.i18n import cn_league

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
OUTCOME_MAP: Dict[str, int] = {
    "home_win": 0,
    "draw": 1,
    "away_win": 2,
}
"""比赛结果到整数标签的映射: 0=主胜, 1=平局, 2=客胜"""

OUTCOME_INV_MAP: Dict[int, str] = {v: k for k, v in OUTCOME_MAP.items()}
"""整数标签到比赛结果的逆向映射"""

MIN_SAMPLES_FOR_LASSO: int = 10
"""LASSO 训练所需的最小样本数,低于此值返回默认权重"""


# ===========================================================================
# 辅助函数
# ===========================================================================

def _validate_input(
    X: np.ndarray,
    y: np.ndarray,
    factor_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    验证并清洗输入数据。

    参数
    ----
    X : np.ndarray, shape (n_matches, n_factors)
        特征矩阵,每行一场比赛,每列一个因子的 delta 值。
    y : np.ndarray, shape (n_matches,)
        标签数组, 0=主胜, 1=平局, 2=客胜。
    factor_names : list of str
        因子 ID 列表,长度必须等于 X 的列数。

    返回
    ----
    X_clean, y_clean, factor_names : 清洗后的数据。

    异常
    ----
    ValueError : 输入维度不匹配或数据为空。
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int32)

    if X.ndim != 2:
        raise ValueError(f"X 必须是二维数组, 当前维度: {X.ndim}")
    if y.ndim != 1:
        raise ValueError(f"y 必须是一维数组, 当前维度: {y.ndim}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X 和 y 的样本数不一致: X={X.shape[0]}, y={y.shape[0]}"
        )
    if X.shape[1] != len(factor_names):
        raise ValueError(
            f"X 的列数({X.shape[1]})与 factor_names 长度({len(factor_names)})不匹配"
        )
    if X.shape[0] == 0:
        raise ValueError("输入数据为空 (0 场比赛)")

    # 剔除包含 NaN 或 Inf 的行
    finite_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if not finite_mask.all():
        n_removed = (~finite_mask).sum()
        logger.warning(
            "检测到 %d 行包含 NaN 或 Inf 值,已剔除", n_removed
        )
        X = X[finite_mask]
        y = y[finite_mask]

    if X.shape[0] == 0:
        raise ValueError("清洗后无有效数据 (所有行均包含 NaN 或 Inf)")

    return X, y, factor_names


def _check_min_samples(
    X: np.ndarray,
    league_id: str,
    min_samples: int = MIN_SAMPLES_FOR_LASSO,
) -> bool:
    """
    检查样本数是否满足 LASSO 训练的最低要求。

    参数
    ----
    X : np.ndarray
        特征矩阵。
    league_id : str
        联赛标识符,用于日志记录。
    min_samples : int
        最低样本数阈值。

    返回
    ----
    bool : True 表示样本数足够, False 表示不足。
    """
    n_samples = X.shape[0]
    if n_samples < min_samples:
        logger.warning(
            "[%s] 样本数 %d < 最低要求 %d, 跳过 LASSO 训练,使用默认权重",
            league_id, n_samples, min_samples,
        )
        return False
    return True


def _check_class_distribution(
    y: np.ndarray,
    league_id: str,
) -> bool:
    """
    检查类别分布是否适合多分类训练 (至少需要 2 个类别)。

    参数
    ----
    y : np.ndarray
        标签数组。
    league_id : str
        联赛标识符。

    返回
    ----
    bool : True 表示类别分布合理, False 表示单一类别。
    """
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        logger.warning(
            "[%s] 仅包含 %d 个类别 %s, 无法训练多分类模型",
            league_id, len(unique_classes), unique_classes.tolist(),
        )
        return False
    return True


def _check_all_zero_columns(
    X: np.ndarray,
    factor_names: List[str],
) -> List[str]:
    """
    检测并报告全零列 (方差为零的特征)。

    全零列不会携带任何信息, LASSO 会将其系数压缩为零。
    此函数仅记录警告,不删除列,因为 StandardScaler 会处理
    零方差特征 (缩放后仍为 0)。

    参数
    ----
    X : np.ndarray
        特征矩阵。
    factor_names : list of str
        因子 ID 列表。

    返回
    ----
    zero_col_factors : 方差为零的因子 ID 列表。
    """
    zero_col_mask = (X.std(axis=0) == 0)
    zero_factors = [factor_names[i] for i in np.where(zero_col_mask)[0]]
    if zero_factors:
        logger.warning(
            "检测到 %d 个全零列因子: %s",
            len(zero_factors), zero_factors,
        )
    return zero_factors


# ===========================================================================
# build_training_data_from_matches
# ===========================================================================

def build_training_data_from_matches(
    matches_data: List[Dict],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    从比赛字典列表构建训练数据 (X, y, factor_names)。

    参数
    ----
    matches_data : list of dict
        比赛数据列表。每个字典必须包含:
        - ``factor_deltas`` : dict[str, dict[str, float]]
            格式: ``{factor_id: {"home": float, "draw": float, "away": float}}``
            每个因子的 delta 值,按 home/draw/away 三个维度。
        - ``actual_outcome`` : str
            实际比赛结果, 取值: ``"home_win"``, ``"draw"``, ``"away_win"``。

    返回
    ----
    X : np.ndarray, shape (n_matches, n_factors)
        特征矩阵。每行一场比赛,每列一个因子。因子值取 home 维度的 delta,
        因为大多数因子的 delta_signs 定义为主队方向
        (home=+1, draw=0, away=-1)。
    y : np.ndarray, shape (n_matches,)
        整数标签: 0=主胜, 1=平局, 2=客胜。
    factor_names : list of str
        因子 ID 列表,按 X 的列顺序排列。

    异常
    ----
    ValueError
        如果 matches_data 为空、因子集不一致或缺少必要字段。

    示例
    ----
    >>> matches = [
    ...     {
    ...         "factor_deltas": {
    ...             "F1": {"home": 0.12, "draw": 0.0, "away": -0.12},
    ...             "F3": {"home": 0.05, "draw": 0.0, "away": -0.05},
    ...         },
    ...         "actual_outcome": "home_win",
    ...     },
    ... ]
    >>> X, y, names = build_training_data_from_matches(matches)
    >>> X.shape
    (1, 2)
    >>> y
    array([0])
    >>> names
    ['F1', 'F3']
    """
    if not matches_data:
        raise ValueError("matches_data 为空,无法构建训练数据")

    # 收集所有比赛中的因子 ID 集合,取并集以确保一致性
    all_factor_ids: set = set()
    for match in matches_data:
        if "factor_deltas" not in match:
            raise ValueError("比赛数据缺少 'factor_deltas' 字段")
        if "actual_outcome" not in match:
            raise ValueError("比赛数据缺少 'actual_outcome' 字段")
        all_factor_ids.update(match["factor_deltas"].keys())

    if not all_factor_ids:
        raise ValueError("未发现任何因子, factor_deltas 为空")

    # 按字母数字排序,保证列顺序稳定
    factor_names = sorted(all_factor_ids, key=lambda x: (
        int(x[1:]) if x[1:].isdigit() else 999, x
    ))
    n_factors = len(factor_names)
    n_matches = len(matches_data)

    X = np.zeros((n_matches, n_factors), dtype=np.float64)
    y = np.zeros(n_matches, dtype=np.int32)

    for i, match in enumerate(matches_data):
        deltas = match["factor_deltas"]
        outcome = match["actual_outcome"]

        if outcome not in OUTCOME_MAP:
            raise ValueError(
                f"未知的比赛结果: '{outcome}', "
                f"允许值: {list(OUTCOME_MAP.keys())}"
            )

        y[i] = OUTCOME_MAP[outcome]

        for j, fid in enumerate(factor_names):
            if fid in deltas:
                # v5.10.7: 使用 max(|home|, |draw|, |away|) 作为特征值
                # 原方案仅取 home 维度会导致 draw-only 因子 (F18/F23/F28)
                # 永远为零列，因为它们的 home delta 恒为 0
                # 新方案: 取三维度最大绝对值，捕获因子在任何维度上的信号强度
                h = deltas[fid].get("home", 0.0)
                d = deltas[fid].get("draw", 0.0)
                a = deltas[fid].get("away", 0.0)
                X[i, j] = max(abs(h), abs(d), abs(a))

    logger.info(
        "从 %d 场比赛构建训练数据: %d 个因子, %d 场比赛",
        n_matches, n_factors, n_matches,
    )

    return X, y, factor_names


# ===========================================================================
# RollingWindowCache
# ===========================================================================

class RollingWindowCache:
    """
    滚动窗口因子选择结果缓存。

    以 ``{league_id}_{window_start}_{window_end}`` 为缓存键,
    存储每个联赛在每个时间窗口内的 LASSO 因子选择结果,
    避免重复计算。

    属性
    ----
    _cache : dict
        内部缓存字典, 键为 ``league_id_windowStart_windowEnd``,
        值为 ``{factor_id: weight}`` 权重字典。
    _timestamps : dict
        记录每个缓存键的插入时间, 用于可选的过期策略。

    用法示例
    --------
    >>> cache = RollingWindowCache()
    >>> key = cache.make_key("premier_league", "2024-01-01", "2024-03-31")
    >>> cache.set(key, {"F1": 1.0, "F2": 0.0})
    >>> cache.get(key)
    {'F1': 1.0, 'F2': 0.0}
    """

    def __init__(self):
        """初始化空缓存。"""
        self._cache: Dict[str, Dict[str, float]] = {}
        self._timestamps: Dict[str, float] = {}

    @staticmethod
    def make_key(league_id: str, window_start: str, window_end: str) -> str:
        """
        生成缓存键。

        参数
        ----
        league_id : str
            联赛标识符, 如 ``"premier_league"``。
        window_start : str
            窗口起始日期, 格式 ``"YYYY-MM-DD"``。
        window_end : str
            窗口结束日期, 格式 ``"YYYY-MM-DD"``。

        返回
        ----
        str : 缓存键, 格式 ``"premier_league_2024-01-01_2024-03-31"``。
        """
        return f"{league_id}_{window_start}_{window_end}"

    def get(self, cache_key: str) -> Optional[Dict[str, float]]:
        """
        获取缓存中的权重字典。

        参数
        ----
        cache_key : str
            缓存键, 由 ``make_key()`` 生成。

        返回
        ----
        dict or None : 如果命中则返回 ``{factor_id: weight}``, 否则返回 None。
        """
        result = self._cache.get(cache_key)
        if result is not None:
            logger.debug("缓存命中: %s", cache_key)
        return result

    def set(self, cache_key: str, weights: Dict[str, float]) -> None:
        """
        将权重字典存入缓存。

        参数
        ----
        cache_key : str
            缓存键。
        weights : dict
            ``{factor_id: weight}`` 权重字典。
        """
        import time
        self._cache[cache_key] = dict(weights)
        self._timestamps[cache_key] = time.time()
        logger.debug("缓存写入: %s (%d 个因子)", cache_key, len(weights))

    def has(self, cache_key: str) -> bool:
        """
        检查缓存键是否存在。

        参数
        ----
        cache_key : str
            缓存键。

        返回
        ----
        bool
        """
        return cache_key in self._cache

    def clear(self) -> None:
        """清空所有缓存。"""
        count = len(self._cache)
        self._cache.clear()
        self._timestamps.clear()
        logger.info("缓存已清空, 共移除 %d 条记录", count)

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, cache_key: str) -> bool:
        return self.has(cache_key)


# ===========================================================================
# RollingLassoSelector
# ===========================================================================

class RollingLassoSelector:
    """
    滚动窗口 LASSO 因子选择器。

    使用 L1 正则化(LASSO)多项逻辑回归,在滚动窗口内自动筛选
    每个联赛的有效因子。对于每个窗口, LASSO 将冗余或噪声因子
    的系数压缩至零,只有非零系数的因子被选中 (权重=1.0),
    其余被剔除 (权重=0.0)。

    属性
    ----
    window_size : int
        滚动窗口大小 (天数), 默认 90 天。
    alpha : float
        LASSO 正则化强度 (C 参数的倒数)。alpha 越大,惩罚越强,
        选中的因子越少。默认 0.01。
    cache : RollingWindowCache
        因子选择结果缓存,按联赛和窗口缓存。
    max_iter : int
        逻辑回归求解器的最大迭代次数。
    random_state : int
        随机种子,用于求解器收敛的确定性。
    _league_weights : dict
        最近一次 ``fit_all()`` 或 ``select()`` 的结果缓存,
        格式: ``{league_id: {factor_id: weight}}``。

    用法示例
    --------
    >>> selector = RollingLassoSelector(window_size=90, alpha=0.01)
    >>> # 单联赛选择
    >>> weights = selector.select("premier_league", X, y, factor_names)
    >>> # 获取已缓存的权重
    >>> cached = selector.get_league_weights("premier_league")
    >>> # 剔除零权重因子
    >>> pruned = selector.prune_weights("premier_league", original_weights)
    """

    def __init__(
        self,
        window_size: int = 90,
        alpha: float = 0.01,
        max_iter: int = 5000,
        random_state: int = 42,
    ):
        """
        初始化滚动窗口 LASSO 选择器。

        参数
        ----
        window_size : int
            滚动窗口大小 (天数)。仅用于记录和日志,实际窗口
            划分由调用方通过传入的 X/y 数据控制。
        alpha : float
            LASSO 正则化强度。 sklearn 的 LogisticRegression
            使用 C 参数, 关系为 C = 1/alpha。
            alpha 越大 -> C 越小 -> 惩罚越强 -> 选中因子越少。
        max_iter : int
            求解器最大迭代次数。 'saga' 求解器可能需要较多
            迭代才能收敛, 默认 5000。
        random_state : int
            随机种子,确保结果可复现。
        """
        self.window_size = window_size
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.cache = RollingWindowCache()
        self._league_weights: Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # 核心方法: select
    # ------------------------------------------------------------------

    def select(
        self,
        league_id: str,
        X: np.ndarray,
        y: np.ndarray,
        factor_names: List[str],
        alpha: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        对指定联赛的数据执行 LASSO 因子选择。

        流程:
        1. 验证输入数据,踢除 NaN/Inf 行。
        2. 检查样本数是否满足最低要求 (>=10)。
        3. 检查类别分布是否合理 (至少 2 个类别)。
        4. 使用 StandardScaler 标准化特征。
        5. 使用 L1 惩罚的多项逻辑回归拟合。
        6. 提取非零系数,生成 {factor_id: 1.0 或 0.0} 权重字典。

        参数
        ----
        league_id : str
            联赛标识符。
        X : np.ndarray, shape (n_matches, n_factors)
            特征矩阵 (因子 delta 值)。
        y : np.ndarray, shape (n_matches,)
            标签数组, 0=主胜, 1=平局, 2=客胜。
        factor_names : list of str
            因子 ID 列表,顺序与 X 的列一致。
        alpha : float, optional
            LASSO 正则化强度。若为 None, 使用实例默认值。

        返回
        ----
        dict[str, float]
            ``{factor_id: weight}``, 选中的因子权重为 1.0,
            被剔除的因子权重为 0.0。

        异常
        ----
        ValueError
            输入数据验证失败。
        """
        if alpha is None:
            alpha = self.alpha

        # 步骤 1: 验证输入
        X, y, factor_names = _validate_input(X, y, factor_names)

        # 步骤 2: 检查最低样本数
        if not _check_min_samples(X, league_id):
            # 样本不足,返回默认权重 (全部为 1.0)
            default_weights = {fid: 1.0 for fid in factor_names}
            self._league_weights[league_id] = default_weights
            return default_weights

        # 步骤 3: 检查类别分布
        if not _check_class_distribution(y, league_id):
            default_weights = {fid: 1.0 for fid in factor_names}
            self._league_weights[league_id] = default_weights
            return default_weights

        # 步骤 4: 检测全零列 (仅记录,不删除)
        _check_all_zero_columns(X, factor_names)

        # 步骤 5: 标准化特征
        # StandardScaler 对零方差列输出 0 (不报错,因为 with_mean=False 时处理)
        # 但对于全零列, with_mean=True 时会在 sklearn >= 1.4 产生警告
        scaler = StandardScaler()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*divide by zero.*",
                category=RuntimeWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=".*invalid value.*",
                category=RuntimeWarning,
            )
            try:
                X_scaled = scaler.fit_transform(X)
            except Exception as e:
                logger.error("[%s] 标准化失败: %s", league_id, e)
                default_weights = {fid: 1.0 for fid in factor_names}
                self._league_weights[league_id] = default_weights
                return default_weights

        # 步骤 6: 训练 LASSO 逻辑回归
        # C = 1/alpha, alpha 越大 -> C 越小 -> 惩罚越强
        C = 1.0 / max(alpha, 1e-10)

        try:
            # sklearn >= 1.5 中 multi_class 已弃用, multinomial 成为默认行为
            # 保留参数以兼容旧版本, 同时抑制 FutureWarning
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*multi_class.*deprecated.*",
                    category=FutureWarning,
                )
                model = LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    multi_class="multinomial",
                    C=C,
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                    tol=1e-4,
                    n_jobs=None,  # 单线程,避免多线程开销
                )
                model.fit(X_scaled, y)
        except Exception as e:
            logger.error(
                "[%s] LASSO 训练失败: %s, 回退至默认权重",
                league_id, e,
            )
            default_weights = {fid: 1.0 for fid in factor_names}
            self._league_weights[league_id] = default_weights
            return default_weights

        # 步骤 7: 提取非零系数
        # 对于 multinomial 多分类, coef_ shape = (n_classes, n_features)
        # 如果任一类别下某因子系数非零,则视为选中
        coef = model.coef_  # shape: (n_classes, n_features)
        n_classes = coef.shape[0]

        if n_classes == 1:
            # 二分类 fallback
            nonzero_mask = np.abs(coef[0]) > 1e-8
        else:
            # 多分类: 任一类别系数非零即选中
            nonzero_mask = (np.abs(coef) > 1e-8).any(axis=0)

        selected_count = int(nonzero_mask.sum())
        total_count = len(factor_names)

        logger.info(
            "[%s] LASSO 选择完成: %d/%d 因子保留 (alpha=%.4f, n=%d)",
            league_id, selected_count, total_count, alpha, X.shape[0],
        )

        # 如果所有因子都被压缩为零,回退到默认权重
        if selected_count == 0:
            logger.warning(
                "[%s] 所有因子系数均为零, 回退至默认权重 (alpha=%.4f 可能过大)",
                league_id, alpha,
            )
            default_weights = {fid: 1.0 for fid in factor_names}
            self._league_weights[league_id] = default_weights
            return default_weights

        # 构建权重字典
        weights = {}
        for j, fid in enumerate(factor_names):
            weights[fid] = 1.0 if nonzero_mask[j] else 0.0

        self._league_weights[league_id] = weights
        return weights

    # ------------------------------------------------------------------
    # get_league_weights
    # ------------------------------------------------------------------

    def get_league_weights(self, league_id: str) -> Dict[str, float]:
        """
        获取最近一次选择或训练后的联赛权重字典。

        参数
        ----
        league_id : str
            联赛标识符。

        返回
        ----
        dict[str, float]
            ``{factor_id: weight}`` 权重字典。

        异常
        ----
        KeyError
            如果该联赛尚未执行过 ``select()`` 或 ``fit_all()``。
        """
        if league_id not in self._league_weights:
            raise KeyError(
                f"联赛 '{league_id}' 尚未执行因子选择, "
                f"请先调用 select() 或 fit_all()"
            )
        return dict(self._league_weights[league_id])

    # ------------------------------------------------------------------
    # prune_weights
    # ------------------------------------------------------------------

    def prune_weights(
        self,
        league_id: str,
        original_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        根据 LASSO 选择结果剔除原始权重中的零权重因子。

        将 ``original_weights`` 中已被 LASSO 剔除的因子权重置零,
        保留被选中的因子权重不变。

        参数
        ----
        league_id : str
            联赛标识符。
        original_weights : dict[str, float]
            原始因子权重字典, 如 ``{"F1": 0.9, "F2": 0.7, ...}``。

        返回
        ----
        dict[str, float]
            剔除后的权重字典。被 LASSO 剔除的因子权重为 0.0,
            保留的因子权重与 original_weights 一致。

        异常
        ----
        KeyError
            如果该联赛尚未执行过因子选择。
        """
        selected = self.get_league_weights(league_id)
        pruned = {}
        for fid, weight in original_weights.items():
            if fid in selected and selected[fid] > 0.0:
                pruned[fid] = weight
            else:
                pruned[fid] = 0.0

        # 确保 selected 中的因子也在 pruned 中 (即使 original 中没有)
        for fid in selected:
            if fid not in pruned:
                pruned[fid] = original_weights.get(fid, 0.0)

        n_pruned = sum(1 for v in pruned.values() if v == 0.0)
        logger.info(
            "[%s] 权重裁剪完成: %d/%d 因子被剔除",
            league_id, n_pruned, len(pruned),
        )
        return pruned

    # ------------------------------------------------------------------
    # fit_all
    # ------------------------------------------------------------------

    def fit_all(
        self,
        training_data: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]],
        alpha: Optional[float] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        对所有联赛批量执行 LASSO 因子选择。

        参数
        ----
        training_data : dict
            按联赛组织的训练数据,格式:
            ``{league_id: (X, y, factor_names)}``
            其中:
            - X: np.ndarray, shape (n_matches, n_factors)
            - y: np.ndarray, shape (n_matches,)
            - factor_names: list of str
        alpha : float, optional
            LASSO 正则化强度。若为 None, 使用实例默认值。

        返回
        ----
        dict[str, dict[str, float]]
            ``{league_id: {factor_id: weight}}`` 所有联赛的权重字典。
        """
        if not training_data:
            raise ValueError("training_data 为空,至少需要一个联赛")

        if alpha is None:
            alpha = self.alpha

        all_weights: Dict[str, Dict[str, float]] = {}

        for league_id, (X, y, factor_names) in sorted(training_data.items()):
            logger.info(
                "[%s] 开始 LASSO 因子选择 (n=%d, p=%d)",
                league_id, X.shape[0], X.shape[1],
            )
            try:
                weights = self.select(league_id, X, y, factor_names, alpha=alpha)
                all_weights[league_id] = weights
            except Exception as e:
                logger.error(
                    "[%s] 因子选择失败: %s, 使用默认权重",
                    league_id, e,
                )
                # 失败时使用默认权重
                all_weights[league_id] = {fid: 1.0 for fid in factor_names}

        logger.info(
            "fit_all 完成: 共处理 %d 个联赛",
            len(all_weights),
        )
        return all_weights

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def get_selected_factors(self, league_id: str) -> List[str]:
        """
        获取指定联赛中被选中的因子 ID 列表。

        参数
        ----
        league_id : str
            联赛标识符。

        返回
        ----
        list[str]
            被选中的因子 ID 列表 (权重为 1.0 的因子)。
        """
        weights = self.get_league_weights(league_id)
        return [fid for fid, w in weights.items() if w > 0.0]

    def get_pruned_factors(self, league_id: str) -> List[str]:
        """
        获取指定联赛中被剔除的因子 ID 列表。

        参数
        ----
        league_id : str
            联赛标识符。

        返回
        ----
        list[str]
            被剔除的因子 ID 列表 (权重为 0.0 的因子)。
        """
        weights = self.get_league_weights(league_id)
        return [fid for fid, w in weights.items() if w == 0.0]

    def summary(self) -> str:
        """
        生成所有联赛的因子选择摘要。

        返回
        ----
        str
            多行文本摘要, 包含每个联赛的选中/剔除因子数量。
        """
        if not self._league_weights:
            return "无因子选择结果 (尚未执行 select() 或 fit_all())"

        lines = ["=== Rolling LASSO 因子选择摘要 ==="]
        for league_id, weights in sorted(self._league_weights.items()):
            selected = sum(1 for w in weights.values() if w > 0.0)
            pruned = sum(1 for w in weights.values() if w == 0.0)
            lines.append(
                f"  {league_id}: {selected} 选中, {pruned} 剔除 "
                f"(共 {len(weights)} 个因子)"
            )
        return "\n".join(lines)


# ===========================================================================
# 模块导出
# ===========================================================================

__all__ = [
    "RollingLassoSelector",
    "RollingWindowCache",
    "build_training_data_from_matches",
]