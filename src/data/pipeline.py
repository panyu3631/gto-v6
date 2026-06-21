"""
GTO-GameFlow v5.1 数据管道

每日定时任务:
1. 拉取联赛积分表 → 更新 standings
2. 拉取赛程/赔率 → 更新 match_results
3. 结算已完赛比赛 → 更新 ELO
4. 时间窗口重置 (每日/每周/每月)
5. 生成资金快照

规范第2.3节: 数据源降级策略
规范第10.10节: 熔断时间窗口重置
"""
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from src.data.api_client import (
    get_source_manager, DataSourceManager,
    FootballDataClient, ApiFootballClient,
)
from src.data.database import (
    get_db, get_repository, DatabaseManager, Repository,
    MatchResult, BetHistory,
)
from src.data.loader import DataLoader
from src.data.models import (
    MatchContext, BetSelection, BetResult, BetPlacement,
)
from src.pipeline.orchestrator import GameFlowPipeline
from src.engine.bankroll import BankrollManager
from src.config.settings import config as global_config

logger = logging.getLogger(__name__)


# ================================================================
# 管道结果
# ================================================================

@dataclass
class PipelineRunResult:
    """单次管道运行结果"""
    run_time: datetime
    leagues_processed: int = 0
    matches_fetched: int = 0
    matches_analyzed: int = 0
    bets_placed: int = 0
    bets_settled: int = 0
    elo_updates: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


# ================================================================
# 每日数据管道
# ================================================================

class DailyDataPipeline:
    """
    每日数据管道 — 编排每日数据获取、分析和投注流程。

    使用方式:
        pipeline = DailyDataPipeline()
        result = pipeline.run()
        print(result.summary())
    """

    SUPPORTED_LEAGUES = [
        "premier_league",
        "la_liga",
        "bundesliga",
        "serie_a",
        "ligue_1",
    ]

    def __init__(
        self,
        db_url: Optional[str] = None,
        initial_bankroll: float = 10000,
    ):
        self.db = get_db()
        self.repo = get_repository()
        self.source_manager = get_source_manager()
        self.bankroll_mgr = BankrollManager(initial_bankroll=initial_bankroll)
        self._last_daily_reset: Optional[datetime] = None
        self._last_weekly_reset: Optional[datetime] = None
        self._last_monthly_reset: Optional[datetime] = None

    # ================================================================
    # 步骤 1: 拉取数据
    # ================================================================

    def fetch_league_data(
        self,
        league_id: str,
        season: str = "2025/26",
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        拉取联赛数据: 积分表 + 赛程。

        返回: (standings, fixtures)
        """
        standings = self.source_manager.get_standings(league_id)

        fixtures = []
        client = self.source_manager.get_available_client("football_data")
        if client and isinstance(client, FootballDataClient):
            try:
                fixtures = client.get_matches(league_id, status="SCHEDULED")
                self.source_manager.record_success("football_data")
            except Exception as e:
                self.source_manager.record_failure("football_data")
                logger.warning(f"football-data.org 获取 {league_id} 赛程失败: {e}")

        # 备用: API-Football
        if not fixtures:
            client2 = self.source_manager.get_available_client("api_football")
            if client2 and isinstance(client2, ApiFootballClient):
                try:
                    fixtures = client2.get_fixtures(league_id, season)
                    self.source_manager.record_success("api_football")
                except Exception as e:
                    self.source_manager.record_failure("api_football")
                    logger.warning(f"API-Football 获取 {league_id} 赛程失败: {e}")

        return standings, fixtures

    def fetch_odds(
        self,
        fixture_id: int,
    ) -> Optional[Dict]:
        """拉取单场比赛赔率"""
        return self.source_manager.get_odds_data(fixture_id)

    # ================================================================
    # 步骤 2: 数据入库
    # ================================================================

    def persist_matches(
        self,
        league_id: str,
        fixtures: List[Dict],
        season: str,
    ) -> int:
        """将赛程数据持久化到数据库"""
        count = 0
        loader = DataLoader(league_id)

        for fixture in fixtures:
            try:
                # 解析 football-data.org 格式
                match_data = self._parse_fixture(fixture, league_id, season)
                if match_data:
                    self.repo.save_match_result(match_data)
                    count += 1
            except Exception as e:
                logger.warning(f"持久化比赛失败: {e}")

        return count

    def _parse_fixture(
        self,
        fixture: Dict,
        league_id: str,
        season: str,
    ) -> Optional[Dict]:
        """解析比赛数据为统一格式"""
        # football-data.org 格式
        if "homeTeam" in fixture:
            home = fixture.get("homeTeam", {})
            away = fixture.get("awayTeam", {})
            odds = fixture.get("odds", {})
            score = fixture.get("score", {}).get("fullTime", {})

            match_data = {
                "match_id": f"{league_id}_{fixture.get('id', '')}",
                "league_id": league_id,
                "season": season,
                "matchday": fixture.get("matchday", 0),
                "kickoff_time": fixture.get("utcDate", datetime.utcnow().isoformat()),
                "home_team": home.get("name", home.get("shortName", "")),
                "away_team": away.get("name", away.get("shortName", "")),
                "home_goals": score.get("home"),
                "away_goals": score.get("away"),
                "is_complete": fixture.get("status") == "FINISHED",
            }

            if odds:
                match_data["odds_home"] = odds.get("homeWin")
                match_data["odds_draw"] = odds.get("draw")
                match_data["odds_away"] = odds.get("awayWin")

            return match_data

        # API-Football 格式
        if "fixture" in fixture:
            f = fixture["fixture"]
            teams = fixture.get("teams", {})
            goals = fixture.get("goals", {})
            odds_data = fixture.get("odds", [])

            match_data = {
                "match_id": f"{league_id}_{f.get('id', '')}",
                "league_id": league_id,
                "season": season,
                "matchday": fixture.get("league", {}).get("round", "0").replace("Regular Season - ", "0"),
                "kickoff_time": f.get("date", datetime.utcnow().isoformat()),
                "home_team": teams.get("home", {}).get("name", ""),
                "away_team": teams.get("away", {}).get("name", ""),
                "home_goals": goals.get("home"),
                "away_goals": goals.get("away"),
                "is_complete": f.get("status", {}).get("short") == "FT",
            }

            return match_data

        return None

    # ================================================================
    # 步骤 3: 分析比赛
    # ================================================================

    def analyze_matches(
        self,
        league_id: str,
        matches: List[MatchResult],
    ) -> List[BetPlacement]:
        """对多场比赛运行流水线分析"""
        placements = []
        pipeline = GameFlowPipeline(league_id, initial_bankroll=self.bankroll_mgr.state.balance)

        # 同步 bankroll 状态
        pipeline.bankroll_mgr.state = self.bankroll_mgr.state

        for match_row in matches:
            try:
                # 构建 MatchContext
                extra_data = self._build_extra_from_db(match_row)
                match = MatchContext(
                    match_id=match_row.match_id,
                    league_id=match_row.league_id,
                    season=match_row.season,
                    matchday=match_row.matchday,
                    kickoff_time=match_row.kickoff_time,
                    home_team=match_row.home_team,
                    away_team=match_row.away_team,
                    home_elo=match_row.home_elo_before or 1500,
                    away_elo=match_row.away_elo_before or 1500,
                    odds_home=match_row.odds_home or 2.0,
                    odds_draw=match_row.odds_draw or 3.5,
                    odds_away=match_row.odds_away or 3.5,
                )

                result = pipeline.run_full(match, extra_data=extra_data)

                if result.placements:
                    placements.extend(result.placements)

                # 持久化因子结果
                if result.factor_deltas:
                    self.repo.save_factor_results(
                        match_row.match_id, league_id, result.factor_deltas
                    )

            except Exception as e:
                logger.error(f"分析比赛 {match_row.match_id} 失败: {e}")

        return placements

    def _build_extra_from_db(self, match_row: MatchResult) -> Dict:
        """从数据库构建 extra_data"""
        league_id = match_row.league_id
        loader = DataLoader(league_id)

        # 获取近期状态
        home_form = self._get_recent_form(match_row.home_team)
        away_form = self._get_recent_form(match_row.away_team)

        # 获取 Elo
        home_elo = self.repo.get_elo(match_row.home_team, league_id)
        away_elo = self.repo.get_elo(match_row.away_team, league_id)

        extra = loader.build_extra_data(
            recent_form=home_form,
            h2h_results=None,
            team_stats=None,
        )

        extra["elo_diff"] = home_elo - away_elo
        extra["data_source_count"] = self._count_available_sources()

        return extra

    def _get_recent_form(self, team_name: str) -> List[float]:
        """获取球队近期状态 (胜=3, 平=1, 负=0)"""
        matches = self.repo.get_match_history(team_name, limit=5)
        form = []
        for m in matches:
            if m.home_goals is None or m.away_goals is None:
                continue
            if m.home_team == team_name:
                if m.home_goals > m.away_goals:
                    form.append(3.0)
                elif m.home_goals == m.away_goals:
                    form.append(1.0)
                else:
                    form.append(0.0)
            else:
                if m.away_goals > m.home_goals:
                    form.append(3.0)
                elif m.away_goals == m.home_goals:
                    form.append(1.0)
                else:
                    form.append(0.0)
        while len(form) < 5:
            form.insert(0, 1.5)
        return form[-5:]

    def _count_available_sources(self) -> int:
        """统计可用数据源数量"""
        status = self.source_manager.check_all_sources()
        return sum(1 for v in status.values() if v)

    # ================================================================
    # 步骤 4: 结算比赛
    # ================================================================

    def settle_completed_matches(self) -> int:
        """结算已完赛的比赛，更新 ELO"""
        count = 0
        pending = self.repo.get_pending_matches()

        for match_row in pending:
            if match_row.home_goals is None or match_row.away_goals is None:
                continue

            # 确定结果
            if match_row.home_goals > match_row.away_goals:
                outcome = "home_win"
                winner = BetSelection.HOME_WIN
            elif match_row.home_goals < match_row.away_goals:
                outcome = "away_win"
                winner = BetSelection.AWAY_WIN
            else:
                outcome = "draw"
                winner = BetSelection.DRAW

            # 结算该比赛的所有投注
            bets = self._get_bets_for_match(match_row.match_id)
            for bet in bets:
                if bet.result != "pending":
                    continue
                if bet.selection == winner.value:
                    profit = bet.stake * (bet.odds - 1.0)
                    self.repo.settle_bet(bet.bet_id, "win", profit)
                    self.bankroll_mgr.settle_bet(
                        BetPlacement(bet.bet_id, bet.match_id, BetSelection(bet.selection),
                                     bet.odds, bet.stake, None),
                        BetResult.WIN, profit,
                    )
                else:
                    self.repo.settle_bet(bet.bet_id, "loss", -bet.stake)
                    self.bankroll_mgr.settle_bet(
                        BetPlacement(bet.bet_id, bet.match_id, BetSelection(bet.selection),
                                     bet.odds, bet.stake, None),
                        BetResult.LOSS, -bet.stake,
                    )

            # 更新 Elo
            home_elo = match_row.home_elo_before or 1500
            away_elo = match_row.away_elo_before or 1500
            new_home, new_away = DataLoader.compute_elos(
                home_elo, away_elo, outcome,
            )
            self.repo.upsert_elo(match_row.home_team, match_row.league_id, new_home)
            self.repo.upsert_elo(match_row.away_team, match_row.league_id, new_away)

            # 标记为完成
            match_row.is_complete = True
            self.repo.save_match_result({
                "match_id": match_row.match_id,
                "is_complete": True,
                "home_elo_before": home_elo,
                "away_elo_before": away_elo,
            })

            count += 1

        return count

    def _get_bets_for_match(self, match_id: str) -> List[BetHistory]:
        """获取某场比赛的投注"""
        with self.db.get_session() as session:
            return session.query(BetHistory).filter_by(match_id=match_id).all()

    # ================================================================
    # 步骤 5: 时间窗口维护
    # ================================================================

    def check_time_windows(self, now: Optional[datetime] = None):
        """
        检查并重置时间窗口。

        规范第10.10节:
        - 每日 00:00 重置 daily_loss
        - 每周一 00:00 重置 weekly_loss
        - 每月 1 日 00:00 重置 monthly_loss
        """
        now = now or datetime.utcnow()

        # 每日重置
        if self._last_daily_reset is None or now.date() > self._last_daily_reset.date():
            self.bankroll_mgr.reset_daily_loss()
            self._last_daily_reset = now
            logger.info("每日亏损窗口已重置")

        # 每周重置 (周一)
        if self._last_weekly_reset is None or (
            now.isocalendar()[1] != self._last_weekly_reset.isocalendar()[1]
        ):
            self.bankroll_mgr.reset_weekly_loss()
            self._last_weekly_reset = now
            logger.info("每周亏损窗口已重置")

        # 每月重置
        if self._last_monthly_reset is None or now.month != self._last_monthly_reset.month:
            self.bankroll_mgr.reset_monthly_loss()
            self._last_monthly_reset = now
            logger.info("每月亏损窗口已重置")

    # ================================================================
    # 主流程
    # ================================================================

    def run(self, season: str = "2025/26") -> PipelineRunResult:
        """
        运行完整每日数据管道。

        流程:
        1. 检查时间窗口
        2. 拉取所有联赛数据
        3. 持久化新赛程
        4. 分析待处理比赛
        5. 结算已完赛比赛
        6. 保存资金快照
        """
        result = PipelineRunResult(run_time=datetime.utcnow())

        try:
            # Step 1: 时间窗口
            self.check_time_windows()

            # Step 2: 拉取数据
            for league_id in self.SUPPORTED_LEAGUES:
                try:
                    standings, fixtures = self.fetch_league_data(league_id, season)
                    result.matches_fetched += len(fixtures)

                    # Step 3: 持久化
                    persisted = self.persist_matches(league_id, fixtures, season)

                    # Step 4: 分析
                    pending = self.repo.get_pending_matches(league_id)
                    placements = self.analyze_matches(league_id, pending)
                    result.matches_analyzed += len(pending)
                    result.bets_placed += len(placements)

                    result.leagues_processed += 1

                except Exception as e:
                    msg = f"处理联赛 {league_id} 失败: {e}"
                    logger.error(msg)
                    result.errors.append(msg)

            # Step 5: 结算
            settled = self.settle_completed_matches()
            result.bets_settled += settled
            result.elo_updates += settled * 2  # 每场更新 2 队

            # Step 6: 保存资金快照
            self.repo.save_bankroll_snapshot({
                "balance": self.bankroll_mgr.state.balance,
                "total_staked": self.bankroll_mgr.state.total_staked,
                "total_returned": self.bankroll_mgr.state.total_returned,
                "total_bets": self.bankroll_mgr.state.total_bets,
                "total_wins": self.bankroll_mgr.state.total_wins,
                "consecutive_losses": self.bankroll_mgr.state.consecutive_losses,
                "max_drawdown": self.bankroll_mgr.state.max_drawdown,
                "snapshot_at": datetime.utcnow(),
            })

        except Exception as e:
            msg = f"管道运行致命错误: {e}"
            logger.error(msg)
            result.errors.append(msg)

        return result

    def summary(self, result: PipelineRunResult) -> str:
        """生成管道运行摘要"""
        lines = []
        lines.append("=" * 60)
        lines.append(f"GTO-GameFlow v5.1 每日管道运行报告")
        lines.append(f"运行时间: {result.run_time.isoformat()}")
        lines.append("-" * 40)
        lines.append(f"联赛处理: {result.leagues_processed}/{len(self.SUPPORTED_LEAGUES)}")
        lines.append(f"比赛拉取: {result.matches_fetched}")
        lines.append(f"比赛分析: {result.matches_analyzed}")
        lines.append(f"投注执行: {result.bets_placed}")
        lines.append(f"比赛结算: {result.bets_settled}")
        lines.append(f"ELO更新: {result.elo_updates}")
        lines.append("-" * 40)
        lines.append(f"当前资金: {self.bankroll_mgr.state.balance:.2f}")
        lines.append(f"总投注: {self.bankroll_mgr.state.total_bets}")
        lines.append(f"总胜场: {self.bankroll_mgr.state.total_wins}")
        lines.append(f"连续亏损: {self.bankroll_mgr.state.consecutive_losses}")
        lines.append(f"ROI: {self.bankroll_mgr.state.roi:.2%}")
        lines.append(f"胜率: {self.bankroll_mgr.state.win_rate:.2%}")
        lines.append(f"最大回撤: {self.bankroll_mgr.state.max_drawdown:.2%}")
        if result.errors:
            lines.append("-" * 40)
            lines.append("错误:")
            for e in result.errors:
                lines.append(f"  - {e}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ================================================================
# 便捷函数
# ================================================================

def run_daily_pipeline(
    db_url: Optional[str] = None,
    season: str = "2025/26",
) -> PipelineRunResult:
    """运行每日管道 (便捷入口)"""
    pipeline = DailyDataPipeline(db_url=db_url)
    result = pipeline.run(season=season)
    print(pipeline.summary(result))
    return result


# ================================================================
# CLI
# ================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("GTO-GameFlow v5.1 数据管道")
    print("=" * 60)

    pipeline = DailyDataPipeline()
    result = pipeline.run()

    print(pipeline.summary(result))

    if result.success:
        print("\n管道运行成功!")
    else:
        print(f"\n管道运行有 {len(result.errors)} 个错误.")