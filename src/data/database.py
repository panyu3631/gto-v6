"""
GTO-GameFlow v5.0 数据层 — 数据库模型与访问

规范第2.5节：持久化策略
- team_elo: 每场比赛后更新
- bankroll: 每次投注结算后
- bet_history: 每次投注结算后追加
- factor_results: 每场因子计算后
"""
import os
import json
from datetime import datetime
from typing import Optional, Dict, List
from sqlalchemy import (
    create_engine, Column, String, Float, Integer, DateTime, JSON,
    Boolean, Text, Index, MetaData,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.pool import StaticPool

from src.config.settings import config as global_config

Base = declarative_base()
metadata = MetaData()


# ================================================================
# 数据库模型
# ================================================================

class TeamElo(Base):
    """球队 Elo 评分表"""
    __tablename__ = "team_elo"

    id = Column(Integer, primary_key=True, autoincrement=True)
    team_name = Column(String(128), nullable=False, index=True)
    league_id = Column(String(64), nullable=False, index=True)
    elo_rating = Column(Float, nullable=False, default=1500.0)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_team_league", "team_name", "league_id", unique=True),
    )


class BetHistory(Base):
    """投注历史表"""
    __tablename__ = "bet_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bet_id = Column(String(128), nullable=False, unique=True, index=True)
    match_id = Column(String(128), nullable=False, index=True)
    league_id = Column(String(64), nullable=False, index=True)
    selection = Column(String(16), nullable=False)  # home_win / draw / away_win
    odds = Column(Float, nullable=False)
    stake = Column(Float, nullable=False)
    placed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    result = Column(String(16), nullable=False, default="pending")  # win / loss / void / pending
    profit_loss = Column(Float, nullable=False, default=0.0)
    settled_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_bet_match", "match_id"),
        Index("idx_bet_placed", "placed_at"),
    )


class BankrollLog(Base):
    """资金日志表"""
    __tablename__ = "bankroll_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    balance = Column(Float, nullable=False)
    total_staked = Column(Float, nullable=False, default=0.0)
    total_returned = Column(Float, nullable=False, default=0.0)
    total_bets = Column(Integer, nullable=False, default=0)
    total_wins = Column(Integer, nullable=False, default=0)
    consecutive_losses = Column(Integer, nullable=False, default=0)
    max_drawdown = Column(Float, nullable=False, default=0.0)
    snapshot_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_bankroll_snapshot", "snapshot_at"),
    )


class FactorResult(Base):
    """因子计算结果表"""
    __tablename__ = "factor_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(128), nullable=False, index=True)
    league_id = Column(String(64), nullable=False, index=True)
    factor_id = Column(String(8), nullable=False, index=True)
    delta_home = Column(Float, nullable=False, default=0.0)
    delta_draw = Column(Float, nullable=False, default=0.0)
    delta_away = Column(Float, nullable=False, default=0.0)
    weight = Column(Float, nullable=False, default=0.0)
    computed_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_factor_match", "match_id", "factor_id"),
    )


class MatchResult(Base):
    """比赛结果表"""
    __tablename__ = "match_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(128), nullable=False, unique=True, index=True)
    league_id = Column(String(64), nullable=False, index=True)
    season = Column(String(16), nullable=False)
    matchday = Column(Integer, nullable=False)
    kickoff_time = Column(DateTime, nullable=False)
    home_team = Column(String(128), nullable=False)
    away_team = Column(String(128), nullable=False)
    home_goals = Column(Integer, nullable=True)
    away_goals = Column(Integer, nullable=True)
    home_elo_before = Column(Float, nullable=True)
    away_elo_before = Column(Float, nullable=True)
    odds_home = Column(Float, nullable=True)
    odds_draw = Column(Float, nullable=True)
    odds_away = Column(Float, nullable=True)
    model_prob_home = Column(Float, nullable=True)
    model_prob_draw = Column(Float, nullable=True)
    model_prob_away = Column(Float, nullable=True)
    is_complete = Column(Boolean, nullable=False, default=False)

    __table_args__ = (
        Index("idx_match_league_season", "league_id", "season"),
    )


# ================================================================
# 数据库管理器
# ================================================================

class DatabaseManager:
    """数据库连接与事务管理 — 支持 PostgreSQL 和 SQLite"""

    def __init__(self, db_url: Optional[str] = None, pool_manager=None):
        self.db_url = db_url or global_config.db_url
        self._engine = None
        self._session_factory = None
        self._pool_manager = pool_manager  # 可选: PGPoolManager 实例

    @property
    def engine(self):
        if self._engine is None:
            # 如果传入了 PGPoolManager，使用它的引擎
            if self._pool_manager is not None:
                self._engine = self._pool_manager.engine
                return self._engine

            connect_args = {}
            kwargs = {
                "echo": False,
                "pool_pre_ping": True,
            }
            # PostgreSQL 参数
            if "postgresql" in self.db_url:
                kwargs["pool_size"] = 5
                kwargs["max_overflow"] = 10
                connect_args.update({
                    "application_name": "gto-gameflow-v5.5",
                    "options": "-c statement_timeout=30000",
                    "keepalives": 1,
                    "keepalives_idle": 30,
                    "keepalives_interval": 10,
                    "keepalives_count": 5,
                })
            # SQLite (测试用)
            elif "sqlite" in self.db_url:
                kwargs["poolclass"] = StaticPool
                connect_args["check_same_thread"] = False

            self._engine = create_engine(
                self.db_url,
                connect_args=connect_args,
                **kwargs,
            )
        return self._engine

    @property
    def session_factory(self):
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        return self._session_factory

    def create_tables(self):
        """创建所有表"""
        Base.metadata.create_all(self.engine)

    def get_session(self) -> Session:
        """获取数据库会话"""
        return self.session_factory()

    def init_db(self):
        """初始化数据库（创建表）"""
        self.create_tables()
        return self

    def health_check(self) -> bool:
        """数据库健康检查"""
        if self._pool_manager is not None:
            return self._pool_manager.health_check()
        try:
            with self.get_session() as session:
                from sqlalchemy import text
                return session.execute(text("SELECT 1")).scalar() == 1
        except Exception:
            return False

    def close(self):
        """关闭数据库连接"""
        if self._pool_manager is not None:
            self._pool_manager.close()
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None


# ================================================================
# 仓库层
# ================================================================

class Repository:
    """数据访问仓库 — 封装所有数据库操作"""

    def __init__(self, db: DatabaseManager):
        self.db = db

    # --- Team Elo ---

    def get_elo(self, team_name: str, league_id: str) -> float:
        """获取球队 Elo 评分"""
        with self.db.get_session() as session:
            row = session.query(TeamElo).filter_by(
                team_name=team_name, league_id=league_id
            ).first()
            return row.elo_rating if row else 1500.0

    def upsert_elo(self, team_name: str, league_id: str, elo_rating: float):
        """更新或插入球队 Elo"""
        with self.db.get_session() as session:
            row = session.query(TeamElo).filter_by(
                team_name=team_name, league_id=league_id
            ).first()
            if row:
                row.elo_rating = elo_rating
                row.updated_at = datetime.utcnow()
            else:
                session.add(TeamElo(
                    team_name=team_name,
                    league_id=league_id,
                    elo_rating=elo_rating,
                    updated_at=datetime.utcnow(),
                ))
            session.commit()

    def batch_upsert_elo(self, entries: List[Dict]):
        """批量更新 Elo"""
        with self.db.get_session() as session:
            for entry in entries:
                row = session.query(TeamElo).filter_by(
                    team_name=entry["team_name"],
                    league_id=entry["league_id"],
                ).first()
                if row:
                    row.elo_rating = entry["elo_rating"]
                    row.updated_at = datetime.utcnow()
                else:
                    session.add(TeamElo(**entry))
            session.commit()

    # --- Bet History ---

    def save_bet(self, bet_data: Dict):
        """保存投注记录"""
        with self.db.get_session() as session:
            session.add(BetHistory(**bet_data))
            session.commit()

    def settle_bet(self, bet_id: str, result: str, profit_loss: float):
        """结算投注"""
        with self.db.get_session() as session:
            row = session.query(BetHistory).filter_by(bet_id=bet_id).first()
            if row:
                row.result = result
                row.profit_loss = profit_loss
                row.settled_at = datetime.utcnow()
                session.commit()

    def get_bet_history(
        self,
        league_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[BetHistory]:
        """获取投注历史"""
        with self.db.get_session() as session:
            q = session.query(BetHistory)
            if league_id:
                q = q.filter_by(league_id=league_id)
            return q.order_by(BetHistory.placed_at.desc()).limit(limit).all()

    # --- Bankroll ---

    def save_bankroll_snapshot(self, snapshot: Dict):
        """保存资金快照"""
        with self.db.get_session() as session:
            session.add(BankrollLog(**snapshot))
            session.commit()

    def get_latest_bankroll(self) -> Optional[BankrollLog]:
        """获取最新资金快照"""
        with self.db.get_session() as session:
            return session.query(BankrollLog).order_by(
                BankrollLog.snapshot_at.desc()
            ).first()

    # --- Factor Results ---

    def save_factor_results(self, match_id: str, league_id: str,
                            factor_deltas: Dict[str, Dict[str, float]]):
        """批量保存因子计算结果"""
        with self.db.get_session() as session:
            for fid, deltas in factor_deltas.items():
                session.add(FactorResult(
                    match_id=match_id,
                    league_id=league_id,
                    factor_id=fid,
                    delta_home=deltas.get("home", 0.0),
                    delta_draw=deltas.get("draw", 0.0),
                    delta_away=deltas.get("away", 0.0),
                    weight=1.0,  # 实际权重由 pipeline 传入
                ))
            session.commit()

    def get_factor_results(self, match_id: str) -> List[FactorResult]:
        """获取某场比赛的因子结果"""
        with self.db.get_session() as session:
            return session.query(FactorResult).filter_by(match_id=match_id).all()

    # --- Match Results ---

    def save_match_result(self, match_data: Dict):
        """保存比赛结果"""
        # 确保日期字段为 datetime 对象
        if "kickoff_time" in match_data and isinstance(match_data["kickoff_time"], str):
            try:
                match_data["kickoff_time"] = datetime.fromisoformat(match_data["kickoff_time"])
            except ValueError:
                from dateutil.parser import parse as date_parse
                try:
                    match_data["kickoff_time"] = date_parse(match_data["kickoff_time"])
                except Exception:
                    match_data["kickoff_time"] = datetime.utcnow()

        with self.db.get_session() as session:
            existing = session.query(MatchResult).filter_by(
                match_id=match_data["match_id"]
            ).first()
            if existing:
                for key, value in match_data.items():
                    setattr(existing, key, value)
            else:
                session.add(MatchResult(**match_data))
            session.commit()

    def get_pending_matches(self, league_id: Optional[str] = None) -> List[MatchResult]:
        """获取待结算比赛"""
        with self.db.get_session() as session:
            q = session.query(MatchResult).filter_by(is_complete=False)
            if league_id:
                q = q.filter_by(league_id=league_id)
            return q.all()

    def get_match_history(
        self,
        team_name: str,
        limit: int = 10,
    ) -> List[MatchResult]:
        """获取球队历史比赛"""
        with self.db.get_session() as session:
            return session.query(MatchResult).filter(
                (MatchResult.home_team == team_name) |
                (MatchResult.away_team == team_name)
            ).order_by(MatchResult.kickoff_time.desc()).limit(limit).all()


# ================================================================
# 全局实例
# ================================================================

_db_manager: Optional[DatabaseManager] = None
_repository: Optional[Repository] = None


def get_db() -> DatabaseManager:
    """获取全局数据库管理器"""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def get_repository() -> Repository:
    """获取全局仓库"""
    global _repository
    if _repository is None:
        _repository = Repository(get_db())
    return _repository


def init_database(db_url: Optional[str] = None):
    """初始化数据库（启动时调用）"""
    global _db_manager, _repository
    _db_manager = DatabaseManager(db_url)
    _db_manager.init_db()
    _repository = Repository(_db_manager)
    return _db_manager, _repository