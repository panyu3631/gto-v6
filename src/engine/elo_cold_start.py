"""
GTO-GameFlow v5.0 Elo 冷启动初始化模块

实现规范文档第3章：通过历史比赛数据构建初始 Elo 评分体系。
冷启动策略:
- 从 CSV 历史数据按时间顺序处理比赛，逐场更新 Elo
- 支持通过 GameFlow 流水线回放历史赛季构建 Elo
- 提供 Elo 估计辅助函数，处理升班马与无数据球队

Elo 更新公式:
    expected = 1.0 / (1.0 + 10.0 ** (-(elo_home + home_adv - elo_away) / 400.0))
    new_elo = old_elo + k * margin_factor * (actual - expected)

其中 margin_factor = 1.0 + min(goal_diff, 3) * 0.33
"""
import csv
import json
import hashlib
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ================================================================
# Elo 初始值估计辅助函数
# ================================================================

def estimate_initial_elo(
    league_id: str,
    team: str,
    historical_data: Optional[Dict[str, Dict[str, Any]]] = None,
    promoted_teams: Optional[Set[str]] = None,
    default_elo: float = 1500.0,
) -> float:
    """
    根据历史数据估计球队的初始 Elo 评分。

    策略优先级:
    1. 若有历史数据且球队存在于历史数据中: 基于历史胜率计算 Elo
       elo = 1500 + 400 * log10(win_rate / (1 - win_rate))
    2. 若球队为升班马 (在 promoted_teams 中或不在历史数据中但其他球队有数据):
       使用 1400-1450 区间，以球队名 hash 确定性取值
    3. 若无任何历史数据: 使用默认值 1500

    Args:
        league_id: 联赛标识符 (如 "premier_league")
        team: 球队名称
        historical_data: 历史数据字典，格式为 {team: {"wins": n, "draws": n, "losses": n}}
                         或 {team: {"win_rate": float}}
        promoted_teams: 已知升班马集合，可选
        default_elo: 默认 Elo 值，当无任何数据时使用

    Returns:
        估计的初始 Elo 评分 (float)
    """
    # 情况 1: 无历史数据 — 使用默认值
    if historical_data is None or len(historical_data) == 0:
        return default_elo

    # 情况 2: 球队存在于历史数据中 — 基于胜率计算
    if team in historical_data:
        stats = historical_data[team]

        # 支持直接提供 win_rate
        if "win_rate" in stats:
            win_rate = stats["win_rate"]
        else:
            wins = stats.get("wins", 0)
            draws = stats.get("draws", 0)
            losses = stats.get("losses", 0)
            total = wins + draws + losses
            if total == 0:
                return default_elo
            # 将平局算作半场胜利，更准确地反映实力
            win_rate = (wins + draws * 0.5) / total

        # 标准 Elo 表现分公式，限制边界避免极端值
        win_rate = max(0.02, min(0.98, win_rate))
        elo = 1500.0 + 400.0 * (__safe_log10(win_rate / (1.0 - win_rate)))
        return round(elo, 1)

    # 情况 3: 球队不在历史数据中 — 可能是升班马
    if promoted_teams is not None and team in promoted_teams:
        is_promoted = True
    else:
        # 若历史数据中有其他球队但无此球队，推断为升班马
        is_promoted = True

    if is_promoted:
        # 使用球队名 hash 在 1400-1450 区间内确定性取值
        seed = int(hashlib.md5(team.encode()).hexdigest(), 16) % 1000
        elo = 1400.0 + (seed / 1000.0) * 50.0
        return round(elo, 1)

    # 兜底: 默认值
    return default_elo


def __safe_log10(x: float) -> float:
    """安全的 log10 计算，避免非正值。"""
    if x <= 0:
        return -3.0  # log10(0.001) ≈ -3
    import math
    return math.log10(x)


# ================================================================
# EloColdStart 类
# ================================================================

class EloColdStart:
    """
    Elo 冷启动初始化器。

    负责从历史比赛数据 (CSV 文件或流水线回放) 构建初始 Elo 评分体系。
    通过按时间顺序逐场处理比赛，模拟 Elo 评分的自然演化过程，
    为后续的实时投注分析提供可靠的球队实力基线。

    使用示例::

        elo_cs = EloColdStart(default_elo=1500.0, k=24, home_advantage=65)
        elo_cs.initialize_from_csv_directory(
            csv_dir="src/data/historical_odds",
            league_ids=["premier_league", "la_liga"],
            seasons=["2014-15", "2015-16", "2016-17"],
        )
        elo_cs.save("/path/to/elo_snapshot.json")
        rating = elo_cs.get_elo("premier_league", "Arsenal")
    """

    def __init__(
        self,
        default_elo: float = 1500.0,
        k: float = 24.0,
        home_advantage: float = 65.0,
    ):
        """
        初始化 EloColdStart 实例。

        Args:
            default_elo: 新球队的默认 Elo 评分
            k: Elo 更新系数 (K-factor)，控制每次比赛后评分的变动幅度
            home_advantage: 主场优势加分，直接加到主队 Elo 上计算预期胜率
        """
        self.default_elo = default_elo
        self.k = k
        self.home_advantage = home_advantage

        # 内部存储: {league_id: {team_name: elo_rating}}
        self._elos: Dict[str, Dict[str, float]] = {}

        # 球队统计: {league_id: {team_name: {wins, draws, losses, goals_for, goals_against, matches}}}
        self._team_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}

        # 已处理的赛季记录
        self._processed_seasons: Dict[str, List[str]] = {}

    # ================================================================
    # 核心方法: 从 CSV 目录初始化
    # ================================================================

    def initialize_from_csv_directory(
        self,
        csv_dir: str,
        league_ids: List[str],
        seasons: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        读取指定联赛和赛季的 CSV 文件，按时间顺序处理所有比赛，
        构建初始 Elo 评分。

        处理流程:
        1. 扫描 csv_dir 下匹配 {league_id}_{season}.csv 模式的文件
        2. 收集所有比赛记录，按日期全局排序
        3. 按时间顺序逐场更新 Elo
        4. 返回最终 Elo 评分字典

        CSV 列要求:
        - Date: 比赛日期 (支持 "%d/%m/%Y" 和 "%d/%m/%y" 格式)
        - HomeTeam: 主队名称
        - AwayTeam: 客队名称
        - FTHG: 全场主队进球数
        - FTAG: 全场客队进球数
        - FTR: 全场结果 (H=主胜, D=平局, A=客胜)

        Args:
            csv_dir: CSV 文件所在目录路径
            league_ids: 联赛标识符列表 (如 ["premier_league", "la_liga"])
            seasons: 赛季列表 (如 ["2014-15", "2015-16", "2016-17"])

        Returns:
            {league_id: {team_name: elo_rating}} 格式的 Elo 评分字典
        """
        all_matches: List[Dict[str, Any]] = []

        # 收集所有比赛记录
        for league_id in league_ids:
            if league_id not in self._elos:
                self._elos[league_id] = {}
            if league_id not in self._team_stats:
                self._team_stats[league_id] = {}
            if league_id not in self._processed_seasons:
                self._processed_seasons[league_id] = []

            for season in seasons:
                file_path = self._find_csv_file(csv_dir, league_id, season)
                if file_path is None:
                    logger.warning(
                        "未找到 CSV 文件: league=%s, season=%s, dir=%s",
                        league_id, season, csv_dir,
                    )
                    continue

                matches = self._parse_csv_file(file_path, league_id, season)
                all_matches.extend(matches)
                self._processed_seasons[league_id].append(season)
                logger.info(
                    "已加载 %d 场比赛: league=%s, season=%s, file=%s",
                    len(matches), league_id, season, os.path.basename(file_path),
                )

        if not all_matches:
            logger.warning("未找到任何比赛数据，所有 Elo 将使用默认值 %.1f", self.default_elo)
            return self._elos

        # 按日期全局排序
        all_matches.sort(key=lambda m: m["date"])

        # 按时间顺序处理每场比赛
        for match in all_matches:
            self._process_single_match(match)

        logger.info(
            "冷启动完成: 处理 %d 场比赛, %d 个联赛, 共 %d 支球队",
            len(all_matches),
            len(league_ids),
            sum(len(teams) for teams in self._elos.values()),
        )

        return self._elos

    def _find_csv_file(
        self,
        csv_dir: str,
        league_id: str,
        season: str,
    ) -> Optional[str]:
        """
        查找匹配 league_id 和 season 的 CSV 文件。

        尝试多种命名模式:
        - {league_id}_{season}.csv
        - {league_id}_{season_long}.csv (如 "2014-2015" 替代 "2014-15")

        Args:
            csv_dir: CSV 文件目录
            league_id: 联赛标识符
            season: 赛季标识符

        Returns:
            找到的文件完整路径，未找到则返回 None
        """
        # 主要模式: premier_league_2014-15.csv
        primary = os.path.join(csv_dir, f"{league_id}_{season}.csv")
        if os.path.isfile(primary):
            return primary

        # 备选模式: 尝试扩展年份格式
        # "2014-15" -> "2014-2015"
        parts = season.split("-")
        if len(parts) == 2 and len(parts[1]) == 2:
            # 推断世纪前缀
            if len(parts[0]) == 4:
                century = parts[0][:2]
                long_season = f"{parts[0]}-{century}{parts[1]}"
                alt = os.path.join(csv_dir, f"{league_id}_{long_season}.csv")
                if os.path.isfile(alt):
                    return alt

        # 反向: "2014-2015" -> "2014-15"
        if len(parts) == 2 and len(parts[1]) == 4:
            short_season = f"{parts[0]}-{parts[1][2:]}"
            alt = os.path.join(csv_dir, f"{league_id}_{short_season}.csv")
            if os.path.isfile(alt):
                return alt

        return None

    def _parse_csv_file(
        self,
        file_path: str,
        league_id: str,
        season: str,
    ) -> List[Dict[str, Any]]:
        """
        解析单个 CSV 文件，返回比赛记录列表。

        支持的日期格式: "%d/%m/%Y" (如 16/08/2014) 和 "%d/%m/%y" (如 16/08/14)

        Args:
            file_path: CSV 文件路径
            league_id: 联赛标识符
            season: 赛季标识符

        Returns:
            比赛记录列表，每条记录包含 date, league_id, season, home_team,
            away_team, fthg, ftag, ftr 字段
        """
        matches = []
        # 支持的日期格式列表
        date_formats = ["%d/%m/%Y", "%d/%m/%y"]

        with open(file_path, "r", encoding="utf-8") as f:
            # 使用 csv.DictReader 自动读取表头
            reader = csv.DictReader(f)

            for row in reader:
                try:
                    # 解析日期
                    date_str = row.get("Date", "").strip()
                    if not date_str:
                        continue

                    parsed_date = None
                    for fmt in date_formats:
                        try:
                            parsed_date = datetime.strptime(date_str, fmt)
                            break
                        except ValueError:
                            continue

                    if parsed_date is None:
                        logger.warning(
                            "无法解析日期 '%s': file=%s, row skipped",
                            date_str, os.path.basename(file_path),
                        )
                        continue

                    # 解析比分
                    fthg_str = row.get("FTHG", "0").strip()
                    ftag_str = row.get("FTAG", "0").strip()
                    try:
                        fthg = int(fthg_str)
                        ftag = int(ftag_str)
                    except ValueError:
                        logger.warning(
                            "无法解析比分 FTHG='%s' FTAG='%s': file=%s, row skipped",
                            fthg_str, ftag_str, os.path.basename(file_path),
                        )
                        continue

                    # 解析结果
                    ftr = row.get("FTR", "").strip().upper()
                    if ftr not in ("H", "D", "A"):
                        logger.warning(
                            "无效的 FTR 值 '%s': file=%s, row skipped",
                            ftr, os.path.basename(file_path),
                        )
                        continue

                    match = {
                        "date": parsed_date,
                        "league_id": league_id,
                        "season": season,
                        "home_team": row.get("HomeTeam", "").strip(),
                        "away_team": row.get("AwayTeam", "").strip(),
                        "fthg": fthg,
                        "ftag": ftag,
                        "ftr": ftr,
                    }

                    if not match["home_team"] or not match["away_team"]:
                        logger.warning(
                            "球队名称为空: file=%s, row skipped",
                            os.path.basename(file_path),
                        )
                        continue

                    matches.append(match)

                except Exception as e:
                    logger.warning(
                        "解析 CSV 行时出错: file=%s, error=%s",
                        os.path.basename(file_path), str(e),
                    )
                    continue

        return matches

    def _process_single_match(self, match: Dict[str, Any]) -> None:
        """
        处理单场比赛，更新两支球队的 Elo 评分和统计数据。

        Elo 更新公式:
            expected = 1.0 / (1.0 + 10.0 ** (-(elo_home + home_adv - elo_away) / 400.0))
            new_elo = old_elo + k * margin_factor * (actual - expected)

        其中 margin_factor = 1.0 + min(goal_diff, 3) * 0.33

        Args:
            match: 比赛记录字典，包含 league_id, home_team, away_team,
                   fthg, ftag, ftr 字段
        """
        league_id = match["league_id"]
        home_team = match["home_team"]
        away_team = match["away_team"]
        fthg = match["fthg"]
        ftag = match["ftag"]
        ftr = match["ftr"]

        # 确保球队 Elo 已初始化
        if league_id not in self._elos:
            self._elos[league_id] = {}
        if league_id not in self._team_stats:
            self._team_stats[league_id] = {}

        # 获取当前 Elo (新球队使用默认值)
        elo_home = self._elos[league_id].get(home_team, self.default_elo)
        elo_away = self._elos[league_id].get(away_team, self.default_elo)

        # 初始化球队统计
        for team in (home_team, away_team):
            if team not in self._team_stats[league_id]:
                self._team_stats[league_id][team] = {
                    "wins": 0, "draws": 0, "losses": 0,
                    "goals_for": 0, "goals_against": 0, "matches": 0,
                }

        # 计算预期主胜概率
        elo_diff = elo_home + self.home_advantage - elo_away
        expected_home = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))

        # 确定实际结果
        if ftr == "H":
            actual_home = 1.0
        elif ftr == "D":
            actual_home = 0.5
        else:  # "A"
            actual_home = 0.0

        # 计算 margin_factor: 大比分胜利带来更大的 Elo 变动
        goal_diff = abs(fthg - ftag)
        margin_factor = 1.0 + min(goal_diff, 3) * 0.33

        # v5.11: 动态 K-factor — 赛季前10轮使用更高K (数据不足时更激进)
        # 第1-5轮: K × 1.4, 第6-10轮: K × 1.2, 第11+轮: K × 1.0
        home_matches = self._team_stats[league_id][home_team]['matches']
        away_matches = self._team_stats[league_id][away_team]['matches']
        avg_matches = (home_matches + away_matches) / 2.0
        if avg_matches < 5:
            k_dynamic = self.k * 1.4
        elif avg_matches < 10:
            k_dynamic = self.k * 1.2
        else:
            k_dynamic = self.k

        # Elo 更新
        delta = k_dynamic * margin_factor * (actual_home - expected_home)
        new_elo_home = elo_home + delta
        new_elo_away = elo_away - delta  # 零和博弈

        # 保存更新后的 Elo
        self._elos[league_id][home_team] = round(new_elo_home, 1)
        self._elos[league_id][away_team] = round(new_elo_away, 1)

        # 更新球队统计
        home_stats = self._team_stats[league_id][home_team]
        away_stats = self._team_stats[league_id][away_team]

        home_stats["matches"] += 1
        away_stats["matches"] += 1
        home_stats["goals_for"] += fthg
        home_stats["goals_against"] += ftag
        away_stats["goals_for"] += ftag
        away_stats["goals_against"] += fthg

        if ftr == "H":
            home_stats["wins"] += 1
            away_stats["losses"] += 1
        elif ftr == "D":
            home_stats["draws"] += 1
            away_stats["draws"] += 1
        else:  # "A"
            home_stats["losses"] += 1
            away_stats["wins"] += 1

    # ================================================================
    # 从历史流水线初始化
    # ================================================================

    def initialize_from_historical_pipeline(
        self,
        league_id: str,
        seasons: List[str],
        pipeline: Any,
        csv_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        通过 GameFlow 流水线回放历史赛季来构建 Elo 评分。

        该方法将历史比赛数据送入流水线逐场处理，从流水线输出中提取
        比赛结果并更新 Elo。如果流水线内部已维护 Elo 状态，则直接同步。

        处理流程:
        1. 从 CSV 文件加载历史比赛数据 (或从 pipeline 获取)
        2. 按时间顺序逐场送入流水线
        3. 根据比赛结果更新 Elo
        4. 返回该联赛的最终 Elo 评分

        Args:
            league_id: 联赛标识符
            seasons: 要回放的赛季列表 (按时间升序)
            pipeline: GameFlowPipeline 实例，必须是已初始化的流水线对象
            csv_dir: CSV 文件目录 (可选，默认尝试从 pipeline 获取数据)

        Returns:
            {team_name: elo_rating} 格式的 Elo 评分字典
        """
        if league_id not in self._elos:
            self._elos[league_id] = {}
        if league_id not in self._team_stats:
            self._team_stats[league_id] = {}
        if league_id not in self._processed_seasons:
            self._processed_seasons[league_id] = []

        total_matches = 0

        for season in seasons:
            # 尝试从 CSV 加载比赛数据
            matches = []
            if csv_dir is not None:
                file_path = self._find_csv_file(csv_dir, league_id, season)
                if file_path is not None:
                    matches = self._parse_csv_file(file_path, league_id, season)
                    logger.info(
                        "流水线回放: 加载 %d 场比赛 (league=%s, season=%s)",
                        len(matches), league_id, season,
                    )
                else:
                    logger.warning(
                        "流水线回放: 未找到 CSV 文件 (league=%s, season=%s)",
                        league_id, season,
                    )

            if not matches:
                # 尝试从 pipeline 获取数据
                logger.info(
                    "流水线回放: 尝试从 pipeline 获取赛季 %s 数据", season,
                )
                try:
                    matches = self._get_matches_from_pipeline(pipeline, league_id, season)
                except Exception as e:
                    logger.warning(
                        "从 pipeline 获取比赛数据失败: %s", str(e),
                    )
                    continue

            if not matches:
                continue

            # 按日期排序
            matches.sort(key=lambda m: m["date"])

            # 逐场通过流水线处理并更新 Elo
            for match in matches:
                try:
                    # 尝试通过流水线处理比赛，获取更丰富的上下文
                    self._run_pipeline_for_match(pipeline, match)
                except Exception as e:
                    logger.debug(
                        "流水线处理比赛失败，使用基础 Elo 更新: %s vs %s, error=%s",
                        match["home_team"], match["away_team"], str(e),
                    )

                # 无论如何都进行基础 Elo 更新
                self._process_single_match(match)
                total_matches += 1

            self._processed_seasons[league_id].append(season)

        logger.info(
            "流水线回放完成: league=%s, 处理 %d 场比赛, %d 支球队",
            league_id, total_matches, len(self._elos.get(league_id, {})),
        )

        return self._elos.get(league_id, {})

    def _get_matches_from_pipeline(
        self,
        pipeline: Any,
        league_id: str,
        season: str,
    ) -> List[Dict[str, Any]]:
        """
        尝试从流水线获取历史比赛数据。

        通过流水线的数据加载能力获取指定联赛和赛季的比赛列表。

        Args:
            pipeline: GameFlowPipeline 实例
            league_id: 联赛标识符
            season: 赛季标识符

        Returns:
            比赛记录列表
        """
        matches = []

        # 尝试通过 pipeline 的 data_loader 或类似接口获取数据
        if hasattr(pipeline, "get_historical_matches"):
            matches = pipeline.get_historical_matches(league_id, season)
        elif hasattr(pipeline, "data_loader") and hasattr(pipeline.data_loader, "load_season"):
            raw_matches = pipeline.data_loader.load_season(league_id, season)
            for rm in raw_matches:
                match = {
                    "date": rm.get("kickoff_time") or rm.get("date"),
                    "league_id": league_id,
                    "season": season,
                    "home_team": rm.get("home_team", ""),
                    "away_team": rm.get("away_team", ""),
                    "fthg": rm.get("home_goals") or rm.get("fthg", 0),
                    "ftag": rm.get("away_goals") or rm.get("ftag", 0),
                    "ftr": rm.get("result") or rm.get("ftr", ""),
                }
                matches.append(match)

        return matches

    def _run_pipeline_for_match(
        self,
        pipeline: Any,
        match: Dict[str, Any],
    ) -> None:
        """
        尝试通过流水线处理单场比赛。

        如果流水线支持，将当前 Elo 注入到比赛上下文中，
        使流水线可以使用最新的 Elo 评分进行计算。

        Args:
            pipeline: GameFlowPipeline 实例
            match: 比赛记录字典
        """
        league_id = match["league_id"]
        home_team = match["home_team"]
        away_team = match["away_team"]

        elo_home = self._elos.get(league_id, {}).get(home_team, self.default_elo)
        elo_away = self._elos.get(league_id, {}).get(away_team, self.default_elo)

        # 尝试通过流水线 process_match 方法处理
        if hasattr(pipeline, "process_match"):
            pipeline.process_match(
                league_id=league_id,
                home_team=home_team,
                away_team=away_team,
                home_elo=elo_home,
                away_elo=elo_away,
                match_date=match["date"],
                result=match["ftr"],
                home_goals=match["fthg"],
                away_goals=match["ftag"],
            )
        elif hasattr(pipeline, "run"):
            # 构建一个最小化的 MatchContext 让流水线运行
            match_ctx = {
                "match_id": f"{league_id}_{match['date'].strftime('%Y%m%d')}_{home_team}_{away_team}",
                "league_id": league_id,
                "home_team": home_team,
                "away_team": away_team,
                "home_elo": elo_home,
                "away_elo": elo_away,
                "season": match.get("season", ""),
            }
            try:
                pipeline.run(match_ctx)
            except Exception:
                pass

    # ================================================================
    # Elo 查询方法
    # ================================================================

    def get_elo(self, league_id: str, team: str) -> float:
        """
        获取指定联赛中某支球队的 Elo 评分。

        Args:
            league_id: 联赛标识符
            team: 球队名称

        Returns:
            Elo 评分，若球队不存在则返回默认值
        """
        return self._elos.get(league_id, {}).get(team, self.default_elo)

    def get_all_elos(self, league_id: str) -> Dict[str, float]:
        """
        获取指定联赛中所有球队的 Elo 评分。

        Args:
            league_id: 联赛标识符

        Returns:
            {team_name: elo_rating} 格式的字典
        """
        return dict(self._elos.get(league_id, {}))

    # ================================================================
    # 持久化: JSON 序列化
    # ================================================================

    def save(self, path: str) -> None:
        """
        将当前 Elo 状态和元数据保存为 JSON 文件。

        保存内容:
        - elos: 所有联赛的球队 Elo 评分
        - team_stats: 球队统计信息
        - processed_seasons: 已处理的赛季列表
        - config: 初始化参数 (default_elo, k, home_advantage)

        Args:
            path: JSON 文件保存路径
        """
        data = {
            "elos": self._elos,
            "team_stats": self._team_stats,
            "processed_seasons": self._processed_seasons,
            "config": {
                "default_elo": self.default_elo,
                "k": self.k,
                "home_advantage": self.home_advantage,
            },
        }

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        logger.info("Elo 状态已保存到: %s", path)

    def load(self, path: str) -> None:
        """
        从 JSON 文件加载 Elo 状态。

        加载后恢复所有内部状态: Elo 评分、球队统计、已处理赛季、
        以及初始化参数。

        Args:
            path: JSON 文件路径

        Raises:
            FileNotFoundError: 文件不存在
            json.JSONDecodeError: JSON 解析失败
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._elos = data.get("elos", {})
        self._team_stats = data.get("team_stats", {})
        self._processed_seasons = data.get("processed_seasons", {})

        # 恢复配置参数
        config = data.get("config", {})
        if config:
            self.default_elo = config.get("default_elo", self.default_elo)
            self.k = config.get("k", self.k)
            self.home_advantage = config.get("home_advantage", self.home_advantage)

        total_teams = sum(len(teams) for teams in self._elos.values())
        logger.info(
            "Elo 状态已从 %s 加载: %d 个联赛, %d 支球队",
            path, len(self._elos), total_teams,
        )

    # ================================================================
    # 辅助方法
    # ================================================================

    def get_team_stats(self, league_id: str, team: str) -> Dict[str, Any]:
        """
        获取指定球队的统计数据。

        Args:
            league_id: 联赛标识符
            team: 球队名称

        Returns:
            包含 wins, draws, losses, goals_for, goals_against, matches 的字典
        """
        default_stats = {
            "wins": 0, "draws": 0, "losses": 0,
            "goals_for": 0, "goals_against": 0, "matches": 0,
        }
        return self._team_stats.get(league_id, {}).get(team, default_stats)

    def get_processed_seasons(self, league_id: str) -> List[str]:
        """
        获取指定联赛已处理的赛季列表。

        Args:
            league_id: 联赛标识符

        Returns:
            已处理赛季的列表
        """
        return list(self._processed_seasons.get(league_id, []))

    def reset(self) -> None:
        """重置所有内部状态，清空所有 Elo 评分和统计数据。"""
        self._elos.clear()
        self._team_stats.clear()
        self._processed_seasons.clear()
        logger.info("EloColdStart 状态已重置")

    def __repr__(self) -> str:
        leagues_count = len(self._elos)
        teams_count = sum(len(teams) for teams in self._elos.values())
        return (
            f"EloColdStart(default_elo={self.default_elo}, k={self.k}, "
            f"home_advantage={self.home_advantage}, "
            f"leagues={leagues_count}, teams={teams_count})"
        )