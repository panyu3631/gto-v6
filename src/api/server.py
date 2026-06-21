"""
GTO v6.0 — API后端

提供RESTful API接口：
1. 比赛数据
2. 预测结果
3. 投注记录
4. 统计数据
5. 配置管理

启动方式:
    python -m src.api.server
    或: python src/api/server.py
"""

from __future__ import annotations
import json
import os
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

# 导入数据库和预测模块
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.database.db_manager import get_database
from src.data.unified_pipeline import get_unified_pipeline
from src.i18n.cn_names import get_cn_name, get_league_cn


class APIHandler(BaseHTTPRequestHandler):
    """API请求处理器"""
    
    def do_GET(self):
        """处理GET请求"""
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)
        
        # 静态文件
        if path == '/' or path == '/index.html':
            self._serve_static('index.html')
            return
        
        # 路由
        if path == '/api/status':
            self._handle_status()
        elif path == '/api/matches':
            self._handle_matches(params)
        elif path == '/api/predictions':
            self._handle_predictions(params)
        elif path == '/api/bets':
            self._handle_bets(params)
        elif path == '/api/stats/roi':
            self._handle_roi_stats(params)
        elif path == '/api/stats/daily':
            self._handle_daily_stats(params)
        elif path == '/api/leagues':
            self._handle_leagues()
        elif path == '/api/factors':
            self._handle_factors()
        elif path.startswith('/api/match/'):
            match_id = path.split('/')[-1]
            self._handle_match_detail(match_id)
        else:
            self._send_error(404, "Not Found")
    
    def do_POST(self):
        """处理POST请求"""
        parsed = urlparse(self.path)
        path = parsed.path
        
        # 读取请求体
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON")
            return
        
        if path == '/api/predict':
            self._handle_predict(data)
        elif path == '/api/bet':
            self._handle_place_bet(data)
        elif path == '/api/config/update':
            self._handle_config_update(data)
        else:
            self._send_error(404, "Not Found")
    
    def _serve_static(self, filename):
        """ serve static files """
        static_dir = os.path.join(os.path.dirname(__file__))
        filepath = os.path.join(static_dir, filename)
        
        if not os.path.exists(filepath):
            self._send_error(404, "File not found")
            return
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(content.encode('utf-8'))
    
    def do_OPTIONS(self):
        """处理CORS预检请求"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def _handle_status(self):
        """系统状态"""
        db = get_database()
        self._send_json({
            "status": "running",
            "version": "6.0",
            "database": "connected",
            "leagues": ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"],
        })
    
    def _handle_matches(self, params: Dict):
        """获取比赛列表"""
        db = get_database()
        league_id = params.get('league_id', [None])[0]
        season = params.get('season', [None])[0]
        limit = int(params.get('limit', [100])[0])
        
        matches = db.get_matches(league_id, season, limit)
        
        # 添加中文名
        for match in matches:
            match['home_team_cn'] = get_cn_name(match.get('home_team', ''))
            match['away_team_cn'] = get_cn_name(match.get('away_team', ''))
            match['league_cn'] = get_league_cn(match.get('league_id', ''))
        
        self._send_json({"matches": matches, "count": len(matches)})
    
    def _handle_match_detail(self, match_id: str):
        """获取比赛详情（完整数据）"""
        from src.database.extended_db import get_extended_database
        from src.data.weather_fetcher import get_weather_fetcher
        from src.data.injury_venue_fetcher import get_injury_fetcher, get_venue_fetcher
        
        db = get_database()
        ext_db = get_extended_database()
        
        match = db.get_match(match_id)
        if not match:
            self._send_error(404, "Match not found")
            return
        
        # 基础信息
        match['home_team_cn'] = get_cn_name(match.get('home_team', ''))
        match['away_team_cn'] = get_cn_name(match.get('away_team', ''))
        match['league_cn'] = get_league_cn(match.get('league_id', ''))
        
        # 多源赔率
        odds = ext_db.get_odds(match_id)
        if odds:
            match['odds_detail'] = odds
        
        # 预测
        predictions = db.get_predictions(match_id)
        match['predictions'] = predictions
        
        # 预测详情（大小球/亚盘/比分矩阵）
        pred_details = ext_db.get_prediction_details(match_id)
        if pred_details:
            match['prediction_details'] = pred_details
        
        # 天气
        weather = ext_db.get_weather(match_id)
        if not weather:
            # 尝试实时获取
            weather_fetcher = get_weather_fetcher()
            city = match.get('home_team', '')
            weather = weather_fetcher.get_weather(city)
            if weather:
                ext_db.insert_weather(match_id, weather)
        match['weather'] = weather
        
        # 伤停
        injuries = ext_db.get_injuries(match_id)
        if not injuries:
            # 尝试实时获取
            injury_fetcher = get_injury_fetcher()
            home_injuries = injury_fetcher.get_injuries(match.get('home_team', ''))
            away_injuries = injury_fetcher.get_injuries(match.get('away_team', ''))
            for inj in home_injuries:
                inj['team'] = 'home'
            for inj in away_injuries:
                inj['team'] = 'away'
            injuries = home_injuries + away_injuries
            if injuries:
                ext_db.insert_injuries(match_id, injuries)
        match['injuries'] = injuries
        
        # 场地
        venue = ext_db.get_venue(match_id)
        if not venue:
            venue_fetcher = get_venue_fetcher()
            venue = venue_fetcher.get_venue(match.get('home_team', ''))
            if venue:
                ext_db.insert_venue(match_id, venue)
        match['venue'] = venue
        
        self._send_json(match)
    
    def _handle_predictions(self, params: Dict):
        """获取预测列表"""
        db = get_database()
        match_id = params.get('match_id', [None])[0]
        limit = int(params.get('limit', [100])[0])
        
        predictions = db.get_predictions(match_id, limit)
        self._send_json({"predictions": predictions, "count": len(predictions)})
    
    def _handle_bets(self, params: Dict):
        """获取投注列表"""
        db = get_database()
        strategy = params.get('strategy', [None])[0]
        league_id = params.get('league_id', [None])[0]
        limit = int(params.get('limit', [100])[0])
        
        bets = db.get_bets(strategy, league_id, limit)
        
        # 添加中文名
        for bet in bets:
            bet['home_team_cn'] = get_cn_name(bet.get('home_team', ''))
            bet['away_team_cn'] = get_cn_name(bet.get('away_team', ''))
            bet['league_cn'] = get_league_cn(bet.get('league_id', ''))
            bet['strategy_cn'] = {
                '1x2': '胜平负',
                'over_under': '大小球',
                'asian_handicap': '亚盘',
            }.get(bet.get('strategy', ''), bet.get('strategy', ''))
        
        self._send_json({"bets": bets, "count": len(bets)})
    
    def _handle_roi_stats(self, params: Dict):
        """获取ROI统计"""
        db = get_database()
        season = params.get('season', [None])[0]
        
        by_league = db.get_roi_by_league(season)
        by_strategy = db.get_roi_by_strategy(season)
        
        # 添加中文名
        for stat in by_league:
            stat['league_cn'] = get_league_cn(stat.get('league_id', ''))
        
        for stat in by_strategy:
            stat['strategy_cn'] = {
                '1x2': '胜平负',
                'over_under': '大小球',
                'asian_handicap': '亚盘',
            }.get(stat.get('strategy', ''), stat.get('strategy', ''))
        
        self._send_json({
            "by_league": by_league,
            "by_strategy": by_strategy,
        })
    
    def _handle_daily_stats(self, params: Dict):
        """获取每日统计"""
        db = get_database()
        days = int(params.get('days', [30])[0])
        
        stats = db.get_daily_stats(days)
        self._send_json({"stats": stats, "count": len(stats)})
    
    def _handle_leagues(self):
        """获取联赛列表"""
        leagues = [
            {"id": "premier_league", "name": "英超", "country": "英格兰"},
            {"id": "la_liga", "name": "西甲", "country": "西班牙"},
            {"id": "bundesliga", "name": "德甲", "country": "德国"},
            {"id": "serie_a", "name": "意甲", "country": "意大利"},
            {"id": "ligue_1", "name": "法甲", "country": "法国"},
        ]
        self._send_json({"leagues": leagues})
    
    def _handle_factors(self):
        """获取因子列表"""
        from src.config.config_loader import get_config
        config = get_config()
        
        factors = []
        for factor_id, factor_config in config._config.get('factors', {}).items():
            factors.append({
                "id": factor_id,
                "name": factor_config.get('name', factor_id),
                "weight": factor_config.get('weight', 0),
                "enabled": factor_config.get('enabled', True),
                "category": factor_config.get('category', ''),
            })
        
        self._send_json({"factors": factors, "count": len(factors)})
    
    def _handle_predict(self, data: Dict):
        """执行预测"""
        try:
            match_id = data.get('match_id', '')
            league_id = data.get('league_id', 'premier_league')
            home_team = data.get('home_team', '')
            away_team = data.get('away_team', '')
            
            # 这里应该调用预测引擎
            # 简化版：返回模拟结果
            prediction = {
                "match_id": match_id,
                "home_team": home_team,
                "away_team": away_team,
                "home_team_cn": get_cn_name(home_team),
                "away_team_cn": get_cn_name(away_team),
                "home_prob": 0.45,
                "draw_prob": 0.25,
                "away_prob": 0.30,
                "recommended_bet": "home",
                "expected_value": 0.05,
                "confidence": 0.6,
            }
            
            # 保存到数据库
            db = get_database()
            db.insert_prediction(prediction)
            
            self._send_json(prediction)
        
        except Exception as e:
            self._send_error(500, str(e))
    
    def _handle_place_bet(self, data: Dict):
        """下注记录"""
        try:
            db = get_database()
            bet_id = db.insert_bet(data)
            
            self._send_json({
                "bet_id": bet_id,
                "status": "recorded",
                "message": "投注已记录",
            })
        
        except Exception as e:
            self._send_error(500, str(e))
    
    def _handle_config_update(self, data: Dict):
        """更新配置"""
        try:
            from src.realtime.hot_updater import get_hot_updater
            updater = get_hot_updater()
            
            update_type = data.get('type')
            
            if update_type == 'factor_weight':
                updater.update_factor_weight(data['factor_id'], data['weight'])
            elif update_type == 'league_param':
                updater.update_league_param(data['league_id'], data['param'], data['value'])
            elif update_type == 'strategy_param':
                updater.update_strategy_param(data['strategy'], data['param'], data['value'])
            
            self._send_json({"status": "updated", "message": "配置已更新"})
        
        except Exception as e:
            self._send_error(500, str(e))
    
    def _send_json(self, data: Any, status: int = 200):
        """发送JSON响应"""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, indent=2).encode('utf-8'))
    
    def _send_error(self, status: int, message: str):
        """发送错误响应"""
        self._send_json({"error": message}, status)
    
    def log_message(self, format, *args):
        """禁用默认日志"""
        pass


def run_server(host: str = '0.0.0.0', port: int = 8080):
    """启动API服务器"""
    server = HTTPServer((host, port), APIHandler)
    print(f"GTO v6.0 API服务器启动: http://{host}:{port}")
    print(f"API文档:")
    print(f"  GET  /api/status          - 系统状态")
    print(f"  GET  /api/matches         - 比赛列表")
    print(f"  GET  /api/match/<id>      - 比赛详情")
    print(f"  GET  /api/predictions     - 预测列表")
    print(f"  GET  /api/bets            - 投注列表")
    print(f"  GET  /api/stats/roi       - ROI统计")
    print(f"  GET  /api/stats/daily     - 每日统计")
    print(f"  GET  /api/leagues         - 联赛列表")
    print(f"  GET  /api/factors         - 因子列表")
    print(f"  POST /api/predict         - 执行预测")
    print(f"  POST /api/bet             - 记录投注")
    print(f"  POST /api/config/update   - 更新配置")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务器已停止")
        server.server_close()


if __name__ == '__main__':
    run_server()
