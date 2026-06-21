"""
GTO v6.0 — SQLite数据库模块

存储历史数据：
1. 比赛记录
2. 预测记录
3. 投注记录
4. 因子数据
"""

from __future__ import annotations
import sqlite3
import json
import os
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'gto.db')


class Database:
    """SQLite数据库管理"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
    
    @contextmanager
    def _get_conn(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _init_db(self):
        """初始化数据库表"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # 比赛记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT UNIQUE,
                    league_id TEXT,
                    season TEXT,
                    home_team TEXT,
                    away_team TEXT,
                    kickoff_time TEXT,
                    home_score INTEGER,
                    away_score INTEGER,
                    result TEXT,
                    home_odds REAL,
                    draw_odds REAL,
                    away_odds REAL,
                    over25_odds REAL,
                    under25_odds REAL,
                    handicap_line REAL,
                    home_elo REAL,
                    away_elo REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 预测记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT,
                    model_version TEXT,
                    home_prob REAL,
                    draw_prob REAL,
                    away_prob REAL,
                    recommended_bet TEXT,
                    recommended_odds REAL,
                    expected_value REAL,
                    confidence REAL,
                    factor_deltas TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (match_id) REFERENCES matches(match_id)
                )
            ''')
            
            # 投注记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS bets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT,
                    prediction_id INTEGER,
                    strategy TEXT,
                    direction TEXT,
                    odds REAL,
                    stake REAL,
                    model_prob REAL,
                    market_prob REAL,
                    value REAL,
                    won BOOLEAN,
                    profit REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (match_id) REFERENCES matches(match_id),
                    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
                )
            ''')
            
            # 因子数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS factor_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT,
                    factor_id TEXT,
                    home_delta REAL,
                    draw_delta REAL,
                    away_delta REAL,
                    weight REAL,
                    enabled BOOLEAN,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (match_id) REFERENCES matches(match_id)
                )
            ''')
            
            # 每日统计表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT UNIQUE,
                    total_bets INTEGER,
                    wins INTEGER,
                    total_staked REAL,
                    total_returned REAL,
                    profit REAL,
                    roi REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 联赛统计表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS league_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    league_id TEXT,
                    season TEXT,
                    total_bets INTEGER,
                    wins INTEGER,
                    total_staked REAL,
                    total_returned REAL,
                    profit REAL,
                    roi REAL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(league_id, season)
                )
            ''')
    
    # 比赛记录操作
    def insert_match(self, match_data: Dict) -> int:
        """插入比赛记录"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO matches 
                (match_id, league_id, season, home_team, away_team, kickoff_time,
                 home_score, away_score, result, home_odds, draw_odds, away_odds,
                 over25_odds, under25_odds, handicap_line, home_elo, away_elo)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                match_data.get('match_id'),
                match_data.get('league_id'),
                match_data.get('season'),
                match_data.get('home_team'),
                match_data.get('away_team'),
                match_data.get('kickoff_time'),
                match_data.get('home_score'),
                match_data.get('away_score'),
                match_data.get('result'),
                match_data.get('home_odds'),
                match_data.get('draw_odds'),
                match_data.get('away_odds'),
                match_data.get('over25_odds'),
                match_data.get('under25_odds'),
                match_data.get('handicap_line'),
                match_data.get('home_elo'),
                match_data.get('away_elo'),
            ))
            return cursor.lastrowid
    
    def get_match(self, match_id: str) -> Optional[Dict]:
        """获取比赛记录"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM matches WHERE match_id = ?', (match_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_matches(self, league_id: str = None, season: str = None, limit: int = 100) -> List[Dict]:
        """获取比赛列表"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            query = 'SELECT * FROM matches WHERE 1=1'
            params = []
            
            if league_id:
                query += ' AND league_id = ?'
                params.append(league_id)
            if season:
                query += ' AND season = ?'
                params.append(season)
            
            query += ' ORDER BY kickoff_time DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    # 预测记录操作
    def insert_prediction(self, prediction_data: Dict) -> int:
        """插入预测记录"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions 
                (match_id, model_version, home_prob, draw_prob, away_prob,
                 recommended_bet, recommended_odds, expected_value, confidence, factor_deltas)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction_data.get('match_id'),
                prediction_data.get('model_version', 'v6.0'),
                prediction_data.get('home_prob'),
                prediction_data.get('draw_prob'),
                prediction_data.get('away_prob'),
                prediction_data.get('recommended_bet'),
                prediction_data.get('recommended_odds'),
                prediction_data.get('expected_value'),
                prediction_data.get('confidence'),
                json.dumps(prediction_data.get('factor_deltas', {})),
            ))
            return cursor.lastrowid
    
    def get_predictions(self, match_id: str = None, limit: int = 100) -> List[Dict]:
        """获取预测列表"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            if match_id:
                cursor.execute('SELECT * FROM predictions WHERE match_id = ? ORDER BY created_at DESC', (match_id,))
            else:
                cursor.execute('SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?', (limit,))
            return [dict(row) for row in cursor.fetchall()]
    
    # 投注记录操作
    def insert_bet(self, bet_data: Dict) -> int:
        """插入投注记录"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO bets 
                (match_id, prediction_id, strategy, direction, odds, stake,
                 model_prob, market_prob, value, won, profit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                bet_data.get('match_id'),
                bet_data.get('prediction_id'),
                bet_data.get('strategy'),
                bet_data.get('direction'),
                bet_data.get('odds'),
                bet_data.get('stake'),
                bet_data.get('model_prob'),
                bet_data.get('market_prob'),
                bet_data.get('value'),
                bet_data.get('won'),
                bet_data.get('profit'),
            ))
            return cursor.lastrowid
    
    def get_bets(self, strategy: str = None, league_id: str = None, limit: int = 100) -> List[Dict]:
        """获取投注列表"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            query = '''
                SELECT b.*, m.league_id, m.home_team, m.away_team 
                FROM bets b 
                JOIN matches m ON b.match_id = m.match_id 
                WHERE 1=1
            '''
            params = []
            
            if strategy:
                query += ' AND b.strategy = ?'
                params.append(strategy)
            if league_id:
                query += ' AND m.league_id = ?'
                params.append(league_id)
            
            query += ' ORDER BY b.created_at DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    # 统计查询
    def get_roi_by_league(self, season: str = None) -> List[Dict]:
        """获取各联赛ROI"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            query = '''
                SELECT m.league_id,
                       COUNT(*) as total_bets,
                       SUM(CASE WHEN b.won THEN 1 ELSE 0 END) as wins,
                       SUM(b.stake) as total_staked,
                       SUM(CASE WHEN b.won THEN b.stake * b.odds ELSE 0 END) as total_returned,
                       SUM(b.profit) as profit
                FROM bets b
                JOIN matches m ON b.match_id = m.match_id
            '''
            params = []
            if season:
                query += ' WHERE m.season = ?'
                params.append(season)
            
            query += ' GROUP BY m.league_id'
            
            cursor.execute(query, params)
            results = []
            for row in cursor.fetchall():
                row_dict = dict(row)
                row_dict['roi'] = row_dict['profit'] / row_dict['total_staked'] if row_dict['total_staked'] > 0 else 0
                row_dict['win_rate'] = row_dict['wins'] / row_dict['total_bets'] if row_dict['total_bets'] > 0 else 0
                results.append(row_dict)
            return results
    
    def get_roi_by_strategy(self, season: str = None) -> List[Dict]:
        """获取各策略ROI"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            query = '''
                SELECT b.strategy,
                       COUNT(*) as total_bets,
                       SUM(CASE WHEN b.won THEN 1 ELSE 0 END) as wins,
                       SUM(b.stake) as total_staked,
                       SUM(CASE WHEN b.won THEN b.stake * b.odds ELSE 0 END) as total_returned,
                       SUM(b.profit) as profit
                FROM bets b
                JOIN matches m ON b.match_id = m.match_id
            '''
            params = []
            if season:
                query += ' WHERE m.season = ?'
                params.append(season)
            
            query += ' GROUP BY b.strategy'
            
            cursor.execute(query, params)
            results = []
            for row in cursor.fetchall():
                row_dict = dict(row)
                row_dict['roi'] = row_dict['profit'] / row_dict['total_staked'] if row_dict['total_staked'] > 0 else 0
                row_dict['win_rate'] = row_dict['wins'] / row_dict['total_bets'] if row_dict['total_bets'] > 0 else 0
                results.append(row_dict)
            return results
    
    def get_daily_stats(self, days: int = 30) -> List[Dict]:
        """获取每日统计"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DATE(b.created_at) as date,
                       COUNT(*) as total_bets,
                       SUM(CASE WHEN b.won THEN 1 ELSE 0 END) as wins,
                       SUM(b.stake) as total_staked,
                       SUM(b.profit) as profit
                FROM bets b
                GROUP BY DATE(b.created_at)
                ORDER BY date DESC
                LIMIT ?
            ''', (days,))
            results = []
            for row in cursor.fetchall():
                row_dict = dict(row)
                row_dict['roi'] = row_dict['profit'] / row_dict['total_staked'] if row_dict['total_staked'] > 0 else 0
                row_dict['win_rate'] = row_dict['wins'] / row_dict['total_bets'] if row_dict['total_bets'] > 0 else 0
                results.append(row_dict)
            return results


# 全局数据库实例
_db = None


def get_database(db_path: str = None) -> Database:
    """获取全局数据库实例"""
    global _db
    if _db is None:
        _db = Database(db_path)
    return _db
