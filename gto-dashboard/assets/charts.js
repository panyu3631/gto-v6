/**
 * GTO-GameFlow v5.5 — 回测仪表盘图表
 * 5 个 ECharts 图表：资金曲线 / ROI 对比 / 投注分布 / 风险回报矩阵 / 胜率与利润因子
 */

(function () {
  'use strict';

  // ─── 主题色 ───────────────────────────────────────────────
  const C = {
    bg: '#0f1923',
    bg2: '#1a2332',
    ink: '#e8ecf1',
    muted: '#7b8ca0',
    rule: '#2a3a4a',
    accent: '#00d4aa',
    accent2: '#f0c040',
    loss: '#ff4757',
    warn: '#ffa502',
  };

  const LEAGUES = ['英超', '西甲', '德甲', '意甲', '法甲'];
  const LEAGUE_COLORS = ['#f0c040', '#00d4aa', '#ff4757', '#ffa502', '#7b8ca0'];
  const SEASON_COLORS = { '23/24': '#00d4aa', '24/25': '#f0c040' };

  // ─── 回测原始数据 ─────────────────────────────────────────
  const DATA = {
    premier_league: {
      '23/24': { bets: 84, wins: 48, winRate: 0.5714, roi: 0.0194, dd: 0.2702, sharpe: 0.0205, pf: 1.0516, final: 10416.38 },
      '24/25': { bets: 143, wins: 77, winRate: 0.5385, roi: -0.1128, dd: 0.3249, sharpe: -0.1127, pf: 0.7514, final: 7594.96 },
    },
    la_liga: {
      '23/24': { bets: 78, wins: 47, winRate: 0.6026, roi: 0.0765, dd: 0.0956, sharpe: 0.0838, pf: 1.2227, final: 10799.81 },
      '24/25': { bets: 112, wins: 67, winRate: 0.5982, roi: 0.0799, dd: 0.0869, sharpe: 0.0877, pf: 1.2372, final: 12121.20 },
    },
    bundesliga: {
      '23/24': { bets: 184, wins: 115, winRate: 0.6250, roi: 0.0981, dd: 0.1041, sharpe: 0.1108, pf: 1.2988, final: 12778.41 },
      '24/25': { bets: 70, wins: 41, winRate: 0.5857, roi: -0.0123, dd: 0.0853, sharpe: -0.0132, pf: 0.9678, final: 12639.76 },
    },
    serie_a: {
      '23/24': { bets: 155, wins: 103, winRate: 0.6645, roi: 0.1097, dd: 0.1315, sharpe: 0.1295, pf: 1.3490, final: 13671.37 },
      '24/25': { bets: 148, wins: 100, winRate: 0.6757, roi: 0.1361, dd: 0.0874, sharpe: 0.1599, pf: 1.4588, final: 18675.29 },
    },
    ligue_1: {
      '23/24': { bets: 47, wins: 31, winRate: 0.6596, roi: 0.0959, dd: 0.0707, sharpe: 0.1101, pf: 1.2924, final: 10629.50 },
      '24/25': { bets: 200, wins: 131, winRate: 0.6550, roi: 0.1247, dd: 0.0451, sharpe: 0.1393, pf: 1.4001, final: 14074.39 },
    },
  };

  const LEAGUE_KEYS = ['premier_league', 'la_liga', 'bundesliga', 'serie_a', 'ligue_1'];

  // ─── 合成资金曲线 ─────────────────────────────────────────
  function generateEquityCurve(numBets, startBalance, finalBalance, volatility, seed) {
    const points = [];
    const totalSteps = 200; // 每日采样点
    const betsPerStep = numBets / totalSteps;
    const targetReturn = (finalBalance - startBalance) / startBalance;
    const drift = targetReturn / totalSteps;
    const sigma = volatility / Math.sqrt(totalSteps);

    let balance = startBalance;
    let running = startBalance;
    points.push({ step: 0, balance: balance });

    // Simple seeded random
    let s = seed || 42;
    function rand() {
      s = (s * 1664525 + 1013904223) & 0x7fffffff;
      return (s >>> 16) / 32767;
    }

    for (let i = 1; i <= totalSteps; i++) {
      const noise = rand() * sigma * 2 - sigma;
      const stepReturn = drift + noise;
      // Simulate bet outcomes: each step has ~betsPerStep bets
      const numBetsThisStep = Math.round(betsPerStep + (rand() - 0.5) * 2);
      for (let j = 0; j < numBetsThisStep; j++) {
        running += (rand() * 200 - 50) * (drift > 0 ? 1 : -0.5);
      }
      // Clamp to reasonable range
      running = Math.max(running * 0.6, Math.min(running * 1.5, running));
      balance = startBalance * (1 + stepReturn * i) + (running - startBalance * (1 + stepReturn * i)) * 0.3;
      balance = Math.max(startBalance * 0.65, Math.min(startBalance * 1.5, balance));
      points.push({ step: i, balance: Math.round(balance * 100) / 100 });
    }
    // Pin final value
    const last = points[points.length - 1];
    const ratio = finalBalance / last.balance;
    for (let i = 0; i < points.length; i++) {
      const t = i / points.length;
      const blend = 1 - Math.pow(1 - t, 3);
      points[i].balance = Math.round((points[i].balance * (1 - blend) + points[i].balance * ratio * blend) * 100) / 100;
    }
    points[points.length - 1].balance = finalBalance;
    return points;
  }

  // 生成日期标签
  function generateDates(startYear, startMonth, numPoints) {
    const dates = [];
    const start = new Date(startYear, startMonth - 1, 15);
    const end = new Date(startYear + 1, 4, 31);
    const totalMs = end - start;
    for (let i = 0; i < numPoints; i++) {
      const d = new Date(start.getTime() + (totalMs * i) / (numPoints - 1));
      dates.push(d.toISOString().slice(0, 10));
    }
    return dates;
  }

  // ─── 全局 ECharts 默认配置 ───────────────────────────────
  function baseOpts() {
    return {
      backgroundColor: 'transparent',
      textStyle: { color: C.muted, fontSize: 11, fontFamily: "-apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif" },
      grid: { top: 20, right: 30, bottom: 35, left: 55 },
      tooltip: {
        backgroundColor: C.bg2,
        borderColor: C.rule,
        textStyle: { color: C.ink, fontSize: 12 },
        extraCssText: 'border-radius:6px;box-shadow:0 4px 20px rgba(0,0,0,0.5);',
      },
      legend: {
        textStyle: { color: C.muted, fontSize: 11 },
        top: 0,
        right: 0,
        itemWidth: 10,
        itemHeight: 10,
        itemGap: 16,
      },
    };
  }

  // ───────────────────────────────────────────────────────────
  // 1. 资金曲线 (chart-equity)
  // ───────────────────────────────────────────────────────────
  (function () {
    const dom = document.getElementById('chart-equity');
    if (!dom) return;
    const chart = echarts.init(dom);

    // 赛季1 综合曲线
    const s1Bets = LEAGUE_KEYS.reduce((s, k) => s + DATA[k]['23/24'].bets, 0);
    const s1Final = LEAGUE_KEYS.reduce((s, k) => s + DATA[k]['23/24'].final, 0);
    const s1Dates = generateDates(2023, 8, 201);
    const s1Curve = generateEquityCurve(s1Bets, 50000, s1Final, 0.18, 123);

    // 赛季2 综合曲线
    const s2Bets = LEAGUE_KEYS.reduce((s, k) => s + DATA[k]['24/25'].bets, 0);
    const s2Final = LEAGUE_KEYS.reduce((s, k) => s + DATA[k]['24/25'].final, 0);
    const s2Dates = generateDates(2024, 8, 201);
    const s2Curve = generateEquityCurve(s2Bets, s1Final, s2Final, 0.14, 456);

    // 赛季1 分联赛曲线
    const leagueS1Series = LEAGUE_KEYS.map((key, idx) => {
      const d = DATA[key]['23/24'];
      const curve = generateEquityCurve(d.bets, 10000, d.final, 0.22, 100 + idx * 50);
      return {
        name: LEAGUES[idx] + ' 23/24',
        type: 'line',
        data: curve.map(p => p.balance),
        lineStyle: { width: 1, color: LEAGUE_COLORS[idx], opacity: 0.6, type: 'dashed' },
        itemStyle: { color: LEAGUE_COLORS[idx] },
        showSymbol: false,
        legendHoverLink: false,
      };
    });

    const option = {
      ...baseOpts(),
      grid: { top: 30, right: 40, bottom: 40, left: 65 },
      xAxis: [
        {
          type: 'category',
          data: s1Dates,
          axisLine: { lineStyle: { color: C.rule } },
          axisTick: { show: false },
          axisLabel: { color: C.muted, fontSize: 10, formatter: v => v.slice(5) },
          splitLine: { show: false },
        },
        {
          type: 'category',
          data: s2Dates,
          axisLine: { lineStyle: { color: C.rule } },
          axisTick: { show: false },
          axisLabel: { color: C.muted, fontSize: 10, formatter: v => v.slice(5) },
          splitLine: { show: false },
          gridIndex: 1,
        },
      ],
      yAxis: [
        {
          type: 'value',
          name: '资金 (¥)',
          nameTextStyle: { color: C.muted, fontSize: 10 },
          axisLabel: { color: C.muted, fontSize: 10, formatter: v => '¥' + (v / 1000).toFixed(0) + 'k' },
          splitLine: { lineStyle: { color: C.rule, type: 'dashed', opacity: 0.4 } },
          min: 42000,
          max: 72000,
        },
        {
          type: 'value',
          gridIndex: 1,
          axisLabel: { color: C.muted, fontSize: 10, formatter: v => '¥' + (v / 1000).toFixed(0) + 'k' },
          splitLine: { lineStyle: { color: C.rule, type: 'dashed', opacity: 0.4 } },
          min: 42000,
          max: 72000,
        },
      ],
      grid: [
        { top: 40, right: 40, bottom: 60, left: 65, height: '35%' },
        { top: '55%', right: 40, bottom: 40, left: 65, height: '35%' },
      ],
      legend: { top: 5, right: 10, textStyle: { color: C.muted, fontSize: 10 } },
      series: [
        // 赛季1 综合
        {
          name: '23/24 赛季',
          xAxisIndex: 0,
          yAxisIndex: 0,
          type: 'line',
          data: s1Curve.map(p => p.balance),
          lineStyle: { width: 2.5, color: C.accent },
          itemStyle: { color: C.accent },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(0,212,170,0.15)' },
              { offset: 1, color: 'rgba(0,212,170,0.01)' },
            ]),
          },
          showSymbol: false,
          smooth: true,
          markLine: {
            silent: true,
            symbol: 'none',
            lineStyle: { color: C.rule, type: 'dashed', width: 1 },
            data: [{ yAxis: 50000, label: { formatter: '初始', color: C.muted, fontSize: 10 } }],
          },
        },
        // 赛季2 综合
        {
          name: '24/25 赛季',
          xAxisIndex: 1,
          yAxisIndex: 1,
          type: 'line',
          data: s2Curve.map(p => p.balance),
          lineStyle: { width: 2.5, color: C.accent2 },
          itemStyle: { color: C.accent2 },
          areaStyle: {
            color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
              { offset: 0, color: 'rgba(240,192,64,0.15)' },
              { offset: 1, color: 'rgba(240,192,64,0.01)' },
            ]),
          },
          showSymbol: false,
          smooth: true,
        },
        // 赛季1 分联赛（虚线）
        ...leagueS1Series,
      ],
      tooltip: {
        ...baseOpts().tooltip,
        formatter: function (params) {
          if (!Array.isArray(params)) params = [params];
          const date = params[0].axisValue || '';
          let html = '<div style="font-weight:600;margin-bottom:4px">' + date + '</div>';
          params.forEach(p => {
            if (p.seriesName.includes('23/24') || p.seriesName.includes('24/25')) {
              html += '<div style="display:flex;justify-content:space-between;gap:20px">'
                + '<span style="color:' + p.color + '">' + p.marker + p.seriesName + '</span>'
                + '<span style="font-weight:600">¥' + (typeof p.value === 'number' ? p.value.toLocaleString() : p.value) + '</span>'
                + '</div>';
            }
          });
          return html;
        },
      },
    };

    chart.setOption(option);
    window.addEventListener('resize', () => chart.resize());
  })();

  // ───────────────────────────────────────────────────────────
  // 2. 联赛 ROI 对比 (chart-roi)
  // ───────────────────────────────────────────────────────────
  (function () {
    const dom = document.getElementById('chart-roi');
    if (!dom) return;
    const chart = echarts.init(dom);

    const s1ROI = LEAGUE_KEYS.map(k => +(DATA[k]['23/24'].roi * 100).toFixed(1));
    const s2ROI = LEAGUE_KEYS.map(k => +(DATA[k]['24/25'].roi * 100).toFixed(1));

    const option = {
      ...baseOpts(),
      grid: { top: 35, right: 30, bottom: 40, left: 55 },
      xAxis: {
        type: 'category',
        data: LEAGUES,
        axisLine: { lineStyle: { color: C.rule } },
        axisTick: { show: false },
        axisLabel: { color: C.ink, fontSize: 12 },
      },
      yAxis: {
        type: 'value',
        name: 'ROI (%)',
        nameTextStyle: { color: C.muted, fontSize: 10 },
        axisLabel: { color: C.muted, fontSize: 10, formatter: '{value}%' },
        splitLine: { lineStyle: { color: C.rule, type: 'dashed', opacity: 0.4 } },
      },
      series: [
        {
          name: '23/24',
          type: 'bar',
          data: s1ROI.map((v, i) => ({
            value: v,
            itemStyle: {
              color: v >= 0 ? new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                { offset: 0, color: C.accent }, { offset: 1, color: 'rgba(0,212,170,0.4)' }
              ]) : new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                { offset: 0, color: C.loss }, { offset: 1, color: 'rgba(255,71,87,0.4)' }
              ]),
              borderRadius: [4, 4, 0, 0],
            },
          })),
          barWidth: '35%',
          barGap: '20%',
          label: { show: true, position: 'top', color: C.muted, fontSize: 10, formatter: '{c}%' },
        },
        {
          name: '24/25',
          type: 'bar',
          data: s2ROI.map((v, i) => ({
            value: v,
            itemStyle: {
              color: v >= 0 ? new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                { offset: 0, color: C.accent2 }, { offset: 1, color: 'rgba(240,192,64,0.4)' }
              ]) : new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                { offset: 0, color: C.loss }, { offset: 1, color: 'rgba(255,71,87,0.4)' }
              ]),
              borderRadius: [4, 4, 0, 0],
            },
          })),
          barWidth: '35%',
          label: { show: true, position: 'top', color: C.muted, fontSize: 10, formatter: '{c}%' },
        },
      ],
      legend: {
        ...baseOpts().legend,
        data: ['23/24', '24/25'],
        top: 0,
      },
    };

    chart.setOption(option);
    window.addEventListener('resize', () => chart.resize());
  })();

  // ───────────────────────────────────────────────────────────
  // 3. 联赛投注分布 (chart-bets)
  // ───────────────────────────────────────────────────────────
  (function () {
    const dom = document.getElementById('chart-bets');
    if (!dom) return;
    const chart = echarts.init(dom);

    const s1Bets = LEAGUE_KEYS.map(k => DATA[k]['23/24'].bets);
    const s2Bets = LEAGUE_KEYS.map(k => DATA[k]['24/25'].bets);

    const option = {
      ...baseOpts(),
      grid: { top: 35, right: 30, bottom: 40, left: 55 },
      xAxis: {
        type: 'category',
        data: LEAGUES,
        axisLine: { lineStyle: { color: C.rule } },
        axisTick: { show: false },
        axisLabel: { color: C.ink, fontSize: 12 },
      },
      yAxis: {
        type: 'value',
        name: '投注数',
        nameTextStyle: { color: C.muted, fontSize: 10 },
        axisLabel: { color: C.muted, fontSize: 10 },
        splitLine: { lineStyle: { color: C.rule, type: 'dashed', opacity: 0.4 } },
      },
      series: [
        {
          name: '23/24',
          type: 'bar',
          data: s1Bets,
          barWidth: '35%',
          barGap: '20%',
          itemStyle: { color: C.accent, borderRadius: [4, 4, 0, 0] },
          label: { show: true, position: 'top', color: C.muted, fontSize: 10 },
        },
        {
          name: '24/25',
          type: 'bar',
          data: s2Bets,
          barWidth: '35%',
          itemStyle: { color: C.accent2, borderRadius: [4, 4, 0, 0] },
          label: { show: true, position: 'top', color: C.muted, fontSize: 10 },
        },
      ],
      legend: {
        ...baseOpts().legend,
        data: ['23/24', '24/25'],
        top: 0,
      },
    };

    chart.setOption(option);
    window.addEventListener('resize', () => chart.resize());
  })();

  // ───────────────────────────────────────────────────────────
  // 4. 风险回报矩阵 (chart-sharpe) — 散点气泡图
  // ───────────────────────────────────────────────────────────
  (function () {
    const dom = document.getElementById('chart-sharpe');
    if (!dom) return;
    const chart = echarts.init(dom);

    const scatterData = [];
    LEAGUE_KEYS.forEach((key, idx) => {
      ['23/24', '24/25'].forEach(season => {
        const d = DATA[key][season];
        const roiPct = d.roi * 100;
        scatterData.push({
          name: LEAGUES[idx] + ' ' + season,
          value: [d.dd * 100, d.sharpe * 100, Math.max(4, Math.abs(roiPct) * 1.2)],
          roi: roiPct,
          itemStyle: { color: LEAGUE_COLORS[idx], opacity: season === '23/24' ? 0.6 : 1 },
          symbolSize: Math.max(8, Math.abs(roiPct) * 1.2),
        });
      });
    });

    const option = {
      ...baseOpts(),
      grid: { top: 35, right: 30, bottom: 45, left: 55 },
      xAxis: {
        type: 'value',
        name: '最大回撤 (%)',
        nameTextStyle: { color: C.muted, fontSize: 10 },
        nameLocation: 'center',
        nameGap: 30,
        axisLabel: { color: C.muted, fontSize: 10, formatter: '{value}%' },
        axisLine: { lineStyle: { color: C.rule } },
        splitLine: { lineStyle: { color: C.rule, type: 'dashed', opacity: 0.4 } },
        inverse: true,
        min: 0,
        max: 40,
      },
      yAxis: {
        type: 'value',
        name: '夏普比率 (×100)',
        nameTextStyle: { color: C.muted, fontSize: 10 },
        nameLocation: 'center',
        nameGap: 40,
        axisLabel: { color: C.muted, fontSize: 10 },
        axisLine: { lineStyle: { color: C.rule } },
        splitLine: { lineStyle: { color: C.rule, type: 'dashed', opacity: 0.4 } },
      },
      series: [
        {
          type: 'scatter',
          data: scatterData,
          emphasis: {
            scale: 1.5,
            label: { show: true, formatter: '{b}', fontSize: 12 },
          },
          label: {
            show: true,
            position: 'right',
            formatter: function (p) {
              return p.name.split(' ')[0];
            },
            color: C.muted,
            fontSize: 9,
          },
          markLine: {
            silent: true,
            symbol: 'none',
            lineStyle: { color: C.rule, type: 'dashed', width: 1, opacity: 0.5 },
            data: [
              { xAxis: 0, label: { show: false } },
              { yAxis: 0, label: { show: false } },
            ],
          },
          markArea: {
            silent: true,
            data: [
              [
                { xAxis: 0, yAxis: 0, itemStyle: { color: 'rgba(0,212,170,0.03)' } },
                { xAxis: 40, yAxis: 20 },
              ],
              [
                { xAxis: 0, yAxis: -20, itemStyle: { color: 'rgba(255,71,87,0.03)' } },
                { xAxis: 40, yAxis: 0 },
              ],
            ],
          },
        },
      ],
      tooltip: {
        ...baseOpts().tooltip,
        formatter: function (p) {
          return '<strong>' + p.name + '</strong><br/>'
            + '回撤: ' + p.value[0].toFixed(1) + '%<br/>'
            + '夏普: ' + (p.value[1] / 100).toFixed(3) + '<br/>'
            + 'ROI: ' + p.data.roi.toFixed(1) + '%';
        },
      },
      legend: { show: false },
    };

    chart.setOption(option);
    window.addEventListener('resize', () => chart.resize());
  })();

  // ───────────────────────────────────────────────────────────
  // 5. 胜率与利润因子 (chart-winrate) — 双轴柱状图
  // ───────────────────────────────────────────────────────────
  (function () {
    const dom = document.getElementById('chart-winrate');
    if (!dom) return;
    const chart = echarts.init(dom);

    const xLabels = [];
    const winRates = [];
    const profitFactors = [];
    const pfColors = [];

    LEAGUE_KEYS.forEach((key, idx) => {
      ['23/24', '24/25'].forEach(season => {
        const d = DATA[key][season];
        xLabels.push(LEAGUES[idx] + '\n' + season);
        winRates.push(+(d.winRate * 100).toFixed(1));
        profitFactors.push(d.pf);
        pfColors.push(d.pf >= 1 ? LEAGUE_COLORS[idx] : C.loss);
      });
    });

    const option = {
      ...baseOpts(),
      grid: { top: 35, right: 60, bottom: 50, left: 55 },
      xAxis: {
        type: 'category',
        data: xLabels,
        axisLine: { lineStyle: { color: C.rule } },
        axisTick: { show: false },
        axisLabel: { color: C.ink, fontSize: 10, interval: 0 },
      },
      yAxis: [
        {
          type: 'value',
          name: '胜率 (%)',
          nameTextStyle: { color: C.muted, fontSize: 10 },
          axisLabel: { color: C.muted, fontSize: 10, formatter: '{value}%' },
          splitLine: { lineStyle: { color: C.rule, type: 'dashed', opacity: 0.4 } },
          min: 40,
          max: 80,
        },
        {
          type: 'value',
          name: '利润因子',
          nameTextStyle: { color: C.muted, fontSize: 10 },
          axisLabel: { color: C.muted, fontSize: 10 },
          splitLine: { show: false },
          min: 0.4,
          max: 2.0,
        },
      ],
      series: [
        {
          name: '胜率',
          type: 'bar',
          data: winRates.map((v, i) => ({
            value: v,
            itemStyle: {
              color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                { offset: 0, color: C.accent }, { offset: 1, color: 'rgba(0,212,170,0.3)' },
              ]),
              borderRadius: [4, 4, 0, 0],
            },
          })),
          barWidth: '55%',
          barGap: '30%',
          label: { show: true, position: 'top', color: C.muted, fontSize: 9, formatter: '{c}%' },
        },
        {
          name: '利润因子',
          type: 'line',
          yAxisIndex: 1,
          data: profitFactors,
          lineStyle: { color: C.accent2, width: 2 },
          itemStyle: { color: C.accent2 },
          symbol: 'circle',
          symbolSize: 8,
          markLine: {
            silent: true,
            symbol: 'none',
            lineStyle: { color: C.rule, type: 'dashed', width: 1 },
            data: [{ yAxis: 1.0, label: { formatter: '盈亏平衡', color: C.muted, fontSize: 9 } }],
          },
        },
      ],
      legend: {
        ...baseOpts().legend,
        data: ['胜率', '利润因子'],
        top: 0,
      },
    };

    chart.setOption(option);
    window.addEventListener('resize', () => chart.resize());
  })();

})();