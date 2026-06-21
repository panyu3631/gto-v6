"""
GTO v6.0 — 数据库扩展

新增字段：
1. 多源赔率（6家博彩公司）
2. 大小球赔率
3. 亚盘赔率
4. 比分矩阵
5. 天气数据
6. 伤停数据
7. 场地信息
"""

from __future__ import annotations
import sqlite3
import json
import os
import logging
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'gto.db')


class ExtendedDatabase:
    """扩展数据库"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_extended_tables()
    
    @contextmanager
    def _get_conn(self):
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
    
    def _init_extended_tables(self):
        """初始化扩展表"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # 扩展比赛表 - 添加多源赔率
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS match_odds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT UNIQUE,
                    
                    -- 1X2 开盘赔率 (6家博彩公司)
                    b365_home REAL, b365_draw REAL, b365_away REAL,
                    bw_home REAL, bw_draw REAL, bw_away REAL,
                    iw_home REAL, iw_draw REAL, iw_away REAL,
                    ps_home REAL, ps_draw REAL, ps_away REAL,
                    wh_home REAL, wh_draw REAL, wh_away REAL,
                    vc_home REAL, vc_draw REAL, vc_away REAL,
                    avg_home REAL, avg_draw REAL, avg_away REAL,
                    max_home REAL, max_draw REAL, max_away REAL,
                    
                    -- 1X2 收盘赔率
                    b365c_home REAL, b365c_draw REAL, b365c_away REAL,
                    psc_home REAL, psc_draw REAL, psc_away REAL,
                    
                    -- 大小球 开盘
                    b365_over25 REAL, b365_under25 REAL,
                    p_over25 REAL, p_under25 REAL,
                    avg_over25 REAL, avg_under25 REAL,
                    max_over25 REAL, max_under25 REAL,
                    
                    -- 大小球 收盘
                    b365c_over25 REAL, b365c_under25 REAL,
                    pc_over25 REAL, pc_under25 REAL,
                    
                    -- 亚盘 开盘
                    ah_line REAL,
                    b365_ah_home REAL, b365_ah_away REAL,
                    ps_ah_home REAL, ps_ah_away REAL,
                    avg_ah_home REAL, avg_ah_away REAL,
                    
                    -- 亚盘 收盘
                    ahc_line REAL,
                    b365c_ah_home REAL, b365c_ah_away REAL,
                    psc_ah_home REAL, psc_ah_away REAL,
                    
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (match_id) REFERENCES matches(match_id)
                )
            ''')
            
            # 扩展预测表 - 添加大小球/亚盘/比分矩阵
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prediction_details (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT,
                    prediction_id INTEGER,
                    
                    -- 大小球概率
                    over25_prob REAL,
                    under25_prob REAL,
                    over15_prob REAL,
                    under15_prob REAL,
                    over35_prob REAL,
                    under35_prob REAL,
                    
                    -- 亚盘概率
                    ah_home_cover REAL,
                    ah_away_cover REAL,
                    ah_void_prob REAL,
                    
                    -- 比分矩阵 (JSON)
                    score_matrix TEXT,
                    
                    -- 预期进球
                    home_lambda REAL,
                    away_lambda REAL,
                    
                    -- 最可能比分
                    most_likely_score TEXT,
                    most_likely_prob REAL,
                    
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (match_id) REFERENCES matches(match_id),
                    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
                )
            ''')
            
            # 天气数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS match_weather (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT UNIQUE,
                    temperature REAL,
                    humidity REAL,
                    wind_speed REAL,
                    precipitation REAL,
                    weather_desc TEXT,
                    weather_impact REAL,
                    fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (match_id) REFERENCES matches(match_id)
                )
            ''')
            
            # 伤停数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS match_injuries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT,
                    team TEXT,
                    player_name TEXT,
                    injury_type TEXT,
                    status TEXT,
                    expected_return TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (match_id) REFERENCES matches(match_id)
                )
            ''')
            
            # 场地信息表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS match_venue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT UNIQUE,
                    venue_name TEXT,
                    city TEXT,
                    country TEXT,
                    capacity INTEGER,
                    surface TEXT,
                    altitude REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (match_id) REFERENCES matches(match_id)
                )
            ''')
    
    # === 赔率操作 ===
    
    def insert_odds(self, match_id: str, odds_data: Dict) -> int:
        """插入多源赔率"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO match_odds 
                (match_id, b365_home, b365_draw, b365_away,
                 bw_home, bw_draw, bw_away,
                 iw_home, iw_draw, iw_away,
                 ps_home, ps_draw, ps_away,
                 wh_home, wh_draw, wh_away,
                 vc_home, vc_draw, vc_away,
                 avg_home, avg_draw, avg_away,
                 max_home, max_draw, max_away,
                 b365_over25, b365_under25,
                 p_over25, p_under25,
                 avg_over25, avg_under25,
                 ah_line, b365_ah_home, b365_ah_away,
                 ps_ah_home, ps_ah_away, avg_ah_home, avg_ah_away)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                match_id,
                odds_data.get('b365_home'), odds_data.get('b365_draw'), odds_data.get('b365_away'),
                odds_data.get('bw_home'), odds_data.get('bw_draw'), odds_data.get('bw_away'),
                odds_data.get('iw_home'), odds_data.get('iw_draw'), odds_data.get('iw_away'),
                odds_data.get('ps_home'), odds_data.get('ps_draw'), odds_data.get('ps_away'),
                odds_data.get('wh_home'), odds_data.get('wh_draw'), odds_data.get('wh_away'),
                odds_data.get('vc_home'), odds_data.get('vc_draw'), odds_data.get('vc_away'),
                odds_data.get('avg_home'), odds_data.get('avg_draw'), odds_data.get('avg_away'),
                odds_data.get('max_home'), odds_data.get('max_draw'), odds_data.get('max_away'),
                odds_data.get('b365_over25'), odds_data.get('b365_under25'),
                odds_data.get('p_over25'), odds_data.get('p_under25'),
                odds_data.get('avg_over25'), odds_data.get('avg_under25'),
                odds_data.get('ah_line'), odds_data.get('b365_ah_home'), odds_data.get('b365_ah_away'),
                odds_data.get('ps_ah_home'), odds_data.get('ps_ah_away'),
                odds_data.get('avg_ah_home'), odds_data.get('avg_ah_away'),
            ))
            return cursor.lastrowid
    
    def get_odds(self, match_id: str) -> Optional[Dict]:
        """获取多源赔率"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM match_odds WHERE match_id = ?', (match_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    # === 预测详情操作 ===
    
    def insert_prediction_details(self, match_id: str, prediction_id: int, details: Dict) -> int:
        """插入预测详情"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO prediction_details
                (match_id, prediction_id, over25_prob, under25_prob,
                 over15_prob, under15_prob, over35_prob, under35_prob,
                 ah_home_cover, ah_away_cover, ah_void_prob,
                 score_matrix, home_lambda, away_lambda,
                 most_likely_score, most_likely_prob)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                match_id, prediction_id,
                details.get('over25_prob'), details.get('under25_prob'),
                details.get('over15_prob'), details.get('under15_prob'),
                details.get('over35_prob'), details.get('under35_prob'),
                details.get('ah_home_cover'), details.get('ah_away_cover'), details.get('ah_void_prob'),
                json.dumps(details.get('score_matrix', {})),
                details.get('home_lambda'), details.get('away_lambda'),
                details.get('most_likely_score'), details.get('most_likely_prob'),
            ))
            return cursor.lastrowid
    
    def get_prediction_details(self, match_id: str) -> Optional[Dict]:
        """获取预测详情"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM prediction_details WHERE match_id = ? ORDER BY created_at DESC LIMIT 1', (match_id,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                if d.get('score_matrix'):
                    d['score_matrix'] = json.loads(d['score_matrix'])
                return d
            return None
    
    # === 天气操作 ===
    
    def insert_weather(self, match_id: str, weather_data: Dict) -> int:
        """插入天气数据"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO match_weather
                (match_id, temperature, humidity, wind_speed, precipitation, 
                 weather_desc, weather_impact)
                VALUES (?,?,?,?,?,?,?)
            ''', (
                match_id,
                weather_data.get('temperature'),
                weather_data.get('humidity'),
                weather_data.get('wind_speed'),
                weather_data.get('precipitation'),
                weather_data.get('weather_desc'),
                weather_data.get('weather_impact'),
            ))
            return cursor.lastrowid
    
    def get_weather(self, match_id: str) -> Optional[Dict]:
        """获取天气数据"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM match_weather WHERE match_id = ?', (match_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    # === 伤停操作 ===
    
    def insert_injuries(self, match_id: str, injuries: List[Dict]) -> int:
        """插入伤停数据"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            # 先删除旧数据
            cursor.execute('DELETE FROM match_injuries WHERE match_id = ?', (match_id,))
            
            count = 0
            for injury in injuries:
                cursor.execute('''
                    INSERT INTO match_injuries
                    (match_id, team, player_name, injury_type, status, expected_return)
                    VALUES (?,?,?,?,?,?)
                ''', (
                    match_id,
                    injury.get('team'),
                    injury.get('player_name'),
                    injury.get('injury_type'),
                    injury.get('status'),
                    injury.get('expected_return'),
                ))
                count += 1
            return count
    
    def get_injuries(self, match_id: str) -> List[Dict]:
        """获取伤停数据"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM match_injuries WHERE match_id = ?', (match_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    # === 场地操作 ===
    
    def insert_venue(self, match_id: str, venue_data: Dict) -> int:
        """插入场地信息"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO match_venue
                (match_id, venue_name, city, country, capacity, surface, altitude)
                VALUES (?,?,?,?,?,?,?)
            ''', (
                match_id,
                venue_data.get('venue_name'),
                venue_data.get('city'),
                venue_data.get('country'),
                venue_data.get('capacity'),
                venue_data.get('surface'),
                venue_data.get('altitude'),
            ))
            return cursor.lastrowid
    
    def get_venue(self, match_id: str) -> Optional[Dict]:
        """获取场地信息"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM match_venue WHERE match_id = ?', (match_id,))
            row = cursor.fetchone()
            return dict(row) if row else None


# 全局实例
_db = None


def get_extended_database(db_path: str = None) -> ExtendedDatabase:
    """获取扩展数据库实例"""
    global _db
    if _db is None:
        _db = ExtendedDatabase(db_path)
    return _db
