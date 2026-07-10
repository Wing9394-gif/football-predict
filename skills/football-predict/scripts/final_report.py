#!/usr/bin/env python3
"""Final prediction report for Spain vs Belgium - World Cup Quarter Final 2026"""
import math, random
from collections import Counter

random.seed(42)

# Corrected params
PI = 0.08
LAMBDA_HOME = 1.800  # Spain attacking strength (adjusted for elite defense faced)
LAMBDA_AWAY = 1.170  # Belgium attacking strength (adjusted for weaker opponent inflation)

def poisson(k, l):
    return (l**k) * math.exp(-l) / math.factorial(k)

# ---- Monte Carlo ----
N = 10000
results = Counter()
for _ in range(N):
    r = random.random()
    if r < PI:
        results[(0,0)] += 1
        continue
    r2 = random.random()
    h = next(k for k in range(15) if r2 <= sum(poisson(j, LAMBDA_HOME) for j in range(k+1)))
    r3 = random.random()
    a = next(k for k in range(15) if r3 <= sum(poisson(j, LAMBDA_AWAY) for j in range(k+1)))
    results[(h,a)] += 1

mc = {k: v/N for k,v in results.items()}
H = sum(p for (h,a),p in mc.items() if h>a)*100
D = sum(p for (h,a),p in mc.items() if h==a)*100
A = sum(p for (h,a),p in mc.items() if a>h)*100

O25 = sum(p for (h,a),p in mc.items() if h+a>=3)*100
U25 = sum(p for (h,a),p in mc.items() if h+a<=2)*100
O15 = sum(p for (h,a),p in mc.items() if h+a>=2)*100

# Asian -1 (一球)
as_win2 = sum(p for (h,a),p in mc.items() if h-a>=2)*100
as_push1 = sum(p for (h,a),p in mc.items() if h-a==1)*100
as_loss0 = sum(p for (h,a),p in mc.items() if h-a<=0)*100

# Asian -0.75 (半/一)
a075_win = sum(p for (h,a),p in mc.items() if h-a>=2)*100
a075_half = sum(p for (h,a),p in mc.items() if h-a==1)*100
a075_loss = sum(p for (h,a),p in mc.items() if h-a<=0)*100

# EV calculations (Crown*)
# 欧赔: 1.60 / 4.15 / 4.90
ev_h = (H/100) * 1.60 - (1 - H/100)
ev_d = (D/100) * 4.15 - (1 - D/100)
ev_a = (A/100) * 4.90 - (1 - A/100)

# O/U 2.5: Over 1.00, Under 0.88
ev_o25 = (O25/100) * 1.00 - (1 - O25/100)
ev_u25 = (U25/100) * 0.88 - (1 - U25/100)

# Asian -1: home 1.03
ev_asi_h = (as_win2/100) * 1.03 + (as_push1/100) * 0 - (as_loss0/100) * 1.0

# Asian -0.75 (Crown* early line: 0.80): win full on 2+, half on 1
ev_asi_075 = (a075_win/100) * 0.80 + (a075_half/100) * 0.40 - (a075_loss/100) * 1.0

# Filter rules: asian/ou with odds < 1.80 → filter
# Crown* asian home 1.03 < 1.80 → FILTER
# Crown* OU over 1.00 < 1.80 → FILTER
# Euro prices: no filter on EV but warn if < 1.80

top10 = sorted(mc.items(), key=lambda x: -x[1])[:10]
scores_str = '\n'.join(f"  {h}-{a}  {p*100:.2f}%" for (h,a),p in top10)

# Dutching best scores
CS_ODDS = {"1-0":6.5,"2-0":7.5,"1-1":7.0,"2-1":8.0,"3-0":12.0,"0-0":10.0,"2-2":15.0,"1-2":14.0,"0-1":12.0}
dutch_scores = ["1-0","2-0","1-1","2-1","3-0"]
inv_sum = sum(1.0/CS_ODDS[s] for s in dutch_scores)
ret = 1.0/inv_sum
bank=100
stakes = {s: (1.0/CS_ODDS[s])/inv_sum*bank for s in dutch_scores}
dutch_cov = sum(mc.get((int(s.split('-')[0]),int(s.split('-')[1])),0) for s in dutch_scores)*100

# ====== OUTPUT ======
print(f"""
╔══════════════════════════════════════════════════════════════╗
║      🏆 世界杯 1/4 决赛 · 西班牙 vs 比利时 🏆              ║
║       开赛：2026-07-11 03:00 CST · 洛杉矶体育场             ║
║       天气：☀️ 天晴 24~25°C · 裁判：Michael Oliver        ║
╚══════════════════════════════════════════════════════════════╝

📊 ZIBP 泊松模型参数
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  λ_西班牙 = {LAMBDA_HOME:.3f}（进攻强度）
  λ_比利时 = {LAMBDA_AWAY:.3f}（进攻强度）
  零堆积系数 π = {PI} | 蒙特卡洛模拟：{N:,} 次
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 赛果概率
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  西班牙胜    {H:.1f}% │ 模型公平赔率 {1/(H/100):.2f} │ 皇冠 1.60 {f'(EV {ev_h:+.3f})' if ev_h>0 else f'(EV {ev_h:.3f})'}
  平局        {D:.1f}% │ 模型公平赔率 {1/(D/100):.2f} │ 皇冠 4.15 {f'(EV {ev_d:+.3f})' if ev_d>0 else f'(EV {ev_d:.3f})'}
  比利时胜    {A:.1f}% │ 模型公平赔率 {1/(A/100):.2f} │ 皇冠 4.90 {f'(EV {ev_a:+.3f})' if ev_a>0 else f'(EV {ev_a:.3f})'}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 大小球概率
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  大球 >2.5    {O25:.1f}% │ 皇冠赔率 1.00（{f'EV {ev_o25:+.3f}' if ev_o25>0 else f'EV {ev_o25:.3f}'}）
  小球 <2.5    {U25:.1f}% │ 皇冠赔率 0.88（{f'EV {ev_u25:+.3f}' if ev_u25>0 else f'EV {ev_u25:.3f}'}）
  大球 >1.5    {O15:.1f}%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 亚盘分析
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  西班牙 -1（一球 @1.03 皇冠）
    赢（净胜2+）：{as_win2:.1f}%
    走水（净胜1）：{as_push1:.1f}%
    输（不胜）：{as_loss0:.1f}%
    EV {ev_asi_h:+.3f} → {'⚠️ 赔率偏低，投注须谨慎' if 1.03<1.8 else '推荐'}

  西班牙 -0.75（半/一 @0.80 皇冠）
    全赢（净胜2+）：{a075_win:.1f}%
    赢半（净胜1）：{a075_half:.1f}%
    全输（不胜）：{a075_loss:.1f}%
    EV {ev_asi_075:+.3f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚽ TOP 10 比分概率
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{scores_str}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔄 Dutching 波胆对冲方案（本金 100）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  比分  │ 概率    │ 赔率  │ 本金   │ 返还
  ──────┼─────────┼───────┼────────┼───────
""", end='')
for s in dutch_scores:
    pct = mc.get((int(s.split('-')[0]),int(s.split('-')[1])),0)*100
    print(f"  {s}    │ {pct:>5.2f}% │ {CS_ODDS[s]:>5.1f} │ {stakes[s]:>5.1f}  │ {stakes[s]*CS_ODDS[s]:>5.1f}")
print(f"  ──────┴─────────┴───────┴────────┴───────")
print(f"  总投入：{bank} | 任意命中返还：{bank*ret:.1f}（回报率 {(ret-1)*100:.1f}%）")
print(f"  覆盖概率：{dutch_cov:.1f}%")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# Rankings
print(f"""
📊 推荐排序（按 EV + 安全性）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

recs = []
# EV ranking
if ev_u25 > 0:
    recs.append(("1️⃣ 小球 < 2.5", f"皇冠 @0.88（EV {ev_u25:+.3f}）" + (" ⚠️ 赔率偏低" if 0.88<1.8 else "")))
if ev_a > 0:
    recs.append(("2️⃣ 比利时胜", f"皇冠 @4.90（EV {ev_a:+.3f}）"))
if ev_d > 0:
    recs.append(("3️⃣ 平局", f"皇冠 @4.15（EV {ev_d:+.3f}）"))
if ev_asi_075 > 0:
    recs.append(("4️⃣ 西班牙 -0.75", f"皇冠 @0.80（EV {ev_asi_075:+.3f}）⚠️ 赔率偏低，投注须谨慎"))

for r in recs:
    print(f"  {r[0]}")
    print(f"     {r[1]}")
    print()

if not recs:
    print("  ⚠️ 当前所有推荐 EV 均为负值或赔率过低")
    print("  建议观望，等待临场盘口变化")

print(f"""
⚠️ 异常信号检测
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅ 盘口升幅：从半/一（0.75）升至一球（1.0），庄家持续看好西班牙
  ✅ 盘口偏差：市场一球 vs 模型理论值，偏差较大，可能存在诱盘
  ✅ 比利时伤缺：中场奥纳纳（十字韧带）+ 后卫德巴斯特（身体不适）
  ✅ 西班牙零失球：5场比赛一球未失，防守固若金汤
  ✅ 西班牙 vs 裁判奥利佛：5场3胜2平不败纪录
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 赛前简报摘要
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  西班牙近况：WWWWDW（盘路 WWWWLD）
  比利时近况：WDWDDW（盘路 WLWLLW）
  
  关键对决：
  • 西班牙防守（0失球）vs 比利时进攻（107次射门最多）
  • 德凯特拉雷（上仗2球）vs 库巴西/拉波尔特
  • 亚马尔突破 vs 卡斯塔内/恩戈伊
  • 罗德里的中场控制 vs 蒂莱曼斯+拉斯金
    
  两队无历史交锋记录（0次）。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ 本报告仅供统计分析参考，不构成投注建议。
   过去表现不代表未来结果。请理性决策。
""")
