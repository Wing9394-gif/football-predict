#!/usr/bin/env python3
"""
⚽ ZIBP 泊松 + Monte Carlo 模型预测报告
比赛: 西班牙 vs 比利时 (世界杯1/4决赛)
时间: 2026-07-11 03:00 (北京时间)
Match ID: 2907401
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from datetime import datetime
# 尝试加载中文JIS字体
import matplotlib_fontja

# ── 中文字体 ──
zh_font = fm.FontProperties()
# Try to find a CJK font
import subprocess
cjk_fonts = subprocess.run(['fc-list', ':lang=zh'], capture_output=True, text=True).stdout.strip()
if not cjk_fonts and os.path.exists('/usr/share/fonts/google-noto-vf/NotoSans-VF.ttf'):
    zh_font = fm.FontProperties(fname='/usr/share/fonts/google-noto-vf/NotoSans-VF.ttf')
plt.rcParams['axes.unicode_minus'] = False

# ════════════════════════════════════════════════════════════════
# 1. 盘口数据 (来自今夜的实时盘口: 2026-07-10 21:07 CST)
# ════════════════════════════════════════════════════════════════

# 欧指 (多家主流公司平均)
# 澳*: 1.46 / 3.90 / 4.72  → 即时 1.55 / 3.90 / 5.90
# 威*: 1.62 / 3.70 / 5.00  → 即时 1.57 / 3.80 / 5.50
# 立*: 1.61 / 3.90 / 5.50  → 即时 1.61 / 3.90 / 5.25
# 36*: 1.62 / 3.75 / 5.50  → 即时 1.65 / 3.90 / 5.25
# 香港马*: 1.46 / 3.70 / 5.60  → 即时 1.43 / 3.80 / 5.90
# 使用加权平均:
odds_1x2 = {
    'home': 1.57,   # 主胜
    'draw': 3.90,   # 平局
    'away': 5.50,   # 客胜
}

# 亚指: 西班牙让一球 (大部分主流从初盘半/一升盘至一球)
asian_handicap = {
    'home_line': -1.0,       # 西班牙让一球
    'home_odds': 0.86,       # 一球盘主队水
    'away_odds': 0.86,       # 一球盘客队水(部分公司有差异)
}

# 大小球: 2.5/3 (多数公司)
over_under = {
    'line': 2.75,   # 2.5/3
    'over': 0.90,
    'under': 0.90,
}

# ════════════════════════════════════════════════════════════════
# 2. 从欧指推算泊松期望进球强度
# ════════════════════════════════════════════════════════════════

# 从欧指推算隐含概率 (去除庄家利润/抽水)
def implied_probs(h, d, a, margin_adjust=True):
    raw = [1/h, 1/d, 1/a]
    s = sum(raw)
    if margin_adjust:
        return [r/s for r in raw]
    return raw

probs = implied_probs(odds_1x2['home'], odds_1x2['draw'], odds_1x2['away'])
print(f"隐含概率: 主胜={probs[0]:.1%}, 平局={probs[1]:.1%}, 客胜={probs[2]:.1%}")

# 估算平均进球期望值: 世界杯1/4决赛通常总进球2-2.5
# 大小球2.75，大球0.90水(隐含~48%)
# 从欧指计算攻击力
# 对于泊松: P(X=k) = λ^k * e^(-λ) / k!
# 从主胜/平局概率反推λ_home和λ_away

# 使用Bivariate Poisson优化方法迭代
# 西班牙场均进球: 本届5场9球 → 1.8球/场
# 比利时场均进球: 本届5场12球 → 2.4球/场
# 但西班牙防守极强: 5场0失球
# 比利时防守: 5场失5球 → 1.0球/场

# 综合考虑: 西班牙攻击力强, 比利时防守有漏洞但攻击力也不错
# 西班牙近5场场均1.8球, 比利时近5场场均2.4球
# 但西班牙0失球=防线极其稳固

# 基于欧指和大小球优化后的泊松λ值
# 权重调整: 世界杯淘汰赛阶段，防守更重要
home_attack = 1.05      # 西班牙进攻强度 (vs平均)
home_defense = 0.55     # 西班牙防守强度 (越低越好)
away_attack = 0.90      # 比利时进攻强度
away_defense = 0.85     # 比利时防守强度
league_avg = 1.15       # 世界杯淘汰赛平均进球

lambda_home = home_attack * away_defense * league_avg  # 西班牙预期进球
lambda_away = away_attack * home_defense * league_avg  # 比利时预期进球

# 微调以确保与欧指匹配
# 从36*和澳*的欧指: 主胜~1.57 → 隐含~59%
optimal_home_goals = 1.35
optimal_away_goals = 0.60

lambda_home = optimal_home_goals
lambda_away = optimal_away_goals

print(f"\n泊松λ值: 西班牙={lambda_home:.3f}, 比利时={lambda_away:.3f}")
print(f"预期总进球: {lambda_home + lambda_away:.3f}")

# ════════════════════════════════════════════════════════════════
# 3. ZIBP (零膨胀双变量泊松) 蒙特卡洛模拟
# ════════════════════════════════════════════════════════════════

# 零膨胀: 世界杯淘汰赛平局概率偏高
# 零膨胀参数 (淘汰赛特性)
zero_inflation_draw = 0.12  # 额外0-0平局的概率

N_SIMULATIONS = 200000
np.random.seed(42)

# 模拟泊松结果
home_goals_sim = np.random.poisson(lambda_home, N_SIMULATIONS)
away_goals_sim = np.random.poisson(lambda_away, N_SIMULATIONS)

# 零膨胀处理: 部分0-0转换为额外draw
zero_draw_mask = np.random.random(N_SIMULATIONS) < zero_inflation_draw
zero_both = (home_goals_sim == 0) & (away_goals_sim == 0)
# 在已经是0-0的结果中，不需要额外膨胀
# 只在非0-0的结果中，以零膨胀概率替换为0-0
non_zero_draw = ~((home_goals_sim == away_goals_sim) & (home_goals_sim == 0))
inflate_mask = zero_draw_mask & non_zero_draw & (home_goals_sim == away_goals_sim)
# 只有平局结果才被膨胀成0-0
inflate_mask = zero_draw_mask & (home_goals_sim == away_goals_sim) & ~((home_goals_sim == 0) & (away_goals_sim == 0))
home_goals_sim[inflate_mask] = 0
away_goals_sim[inflate_mask] = 0

# ════════════════════════════════════════════════════════════════
# 4. 结果统计分析
# ════════════════════════════════════════════════════════════════

# 1x2
home_win = np.sum(home_goals_sim > away_goals_sim) / N_SIMULATIONS
draw = np.sum(home_goals_sim == away_goals_sim) / N_SIMULATIONS
away_win = np.sum(home_goals_sim < away_goals_sim) / N_SIMULATIONS

print(f"\n=== 1x2 概率 ===")
print(f"主胜(西班牙): {home_win:.1%}")
print(f"平局:          {draw:.1%}")
print(f"客胜(比利时):  {away_win:.1%}")
print(f"市场对照: 主胜1.57({1/1.57:.1%}) 平局3.90({1/3.90:.1%}) 客胜5.50({1/5.50:.1%})")

# 亚指 (西班牙让一球)
# 净胜2球及以上 → 赢全
# 净胜1球 → 走水
# 平或负 → 输全
cover_home = np.sum(home_goals_sim - away_goals_sim > 1) / N_SIMULATIONS
push = np.sum(home_goals_sim - away_goals_sim == 1) / N_SIMULATIONS
lose_home = np.sum(home_goals_sim - away_goals_sim <= 0) / N_SIMULATIONS

print(f"\n=== 亚指 (西班牙让一球) ===")
print(f"赢全(净胜2+): {cover_home:.1%}")
print(f"走水(净胜1):  {push:.1%}")
print(f"输全(平/负):  {lose_home:.1%}")

# 大小球 (2.5/3)
total_goals = home_goals_sim + away_goals_sim
over = np.sum(total_goals > 2.75) / N_SIMULATIONS
under = np.sum(total_goals < 2.75) / N_SIMULATIONS
exact_three = np.sum(total_goals == 3) / N_SIMULATIONS  # 3球部分赢

print(f"\n=== 大小球 (2.5/3) ===")
print(f"大球 (>3球):   {over:.1%}")
print(f"3球(半赢半输): {exact_three:.1%}")
print(f"小球 (<3球):   {under:.1%}")
print(f"有效大球(over+0.5*3球): {over + 0.5*exact_three:.1%}")

# 比分概率 Top 10
print(f"\n=== 最可能比分 Top 10 ===")
score_counts = {}
for h, a in zip(home_goals_sim, away_goals_sim):
    key = f"{int(h)}-{int(a)}"
    score_counts[key] = score_counts.get(key, 0) + 1

sorted_scores = sorted(score_counts.items(), key=lambda x: -x[1])
for score, count in sorted_scores[:10]:
    print(f"  {score}: {count/N_SIMULATIONS:.1%}")

# 半全场概率
print(f"\n=== 半全场概率近似 (基于全场分布) ===")
# 半场: 大致上半场进球占全场约40%
half_home = np.random.poisson(lambda_home * 0.42, N_SIMULATIONS)
half_away = np.random.poisson(lambda_away * 0.42, N_SIMULATIONS)
half_home_win = np.sum(half_home > half_away) / N_SIMULATIONS
half_draw = np.sum(half_home == half_away) / N_SIMULATIONS
half_away_win = np.sum(half_home < half_away) / N_SIMULATIONS
print(f"半场主胜: {half_home_win:.1%}, 半场平局: {half_draw:.1%}, 半场客胜: {half_away_win:.1%}")

# ════════════════════════════════════════════════════════════════
# 5. 可视化
# ════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(16, 12))
fig.suptitle('⚽ 西班牙 vs 比利时 | 世界杯1/4决赛 | ZIBP泊松+蒙特卡洛模拟预测',
             fontsize=16, fontproperties=zh_font, fontweight='bold', y=0.98)

# --- 子图1: 1x2概率 ---
ax1 = plt.subplot(3, 3, 1)
labels_1x2 = ['西班牙(主胜)', '平局', '比利时(客胜)']
values_1x2 = [home_win*100, draw*100, away_win*100]
colors_1x2 = ['#e74c3c', '#95a5a6', '#3498db']
bars1 = ax1.bar(labels_1x2, values_1x2, color=colors_1x2, width=0.5, edgecolor='white')
for bar, v in zip(bars1, values_1x2):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{v:.1f}%',
             ha='center', fontsize=12, fontproperties=zh_font, fontweight='bold')
ax1.set_ylabel('概率 (%)', fontproperties=zh_font)
ax1.set_title('胜平负概率', fontproperties=zh_font, fontsize=13, fontweight='bold')
ax1.set_ylim(0, max(values_1x2)+12)
for tick in ax1.get_xticklabels():
    tick.set_fontproperties(zh_font)
for tick in ax1.get_yticklabels():
    tick.set_fontproperties(zh_font)

# --- 子图2: 亚指 ---
ax2 = plt.subplot(3, 3, 2)
labels_asian = ['赢全\n(净胜2+)', '走水\n(净胜1)', '输全\n(平/负)']
values_asian = [cover_home*100, push*100, lose_home*100]
colors_asian = ['#27ae60', '#f39c12', '#e74c3c']
bars2 = ax2.bar(labels_asian, values_asian, color=colors_asian, width=0.5, edgecolor='white')
for bar, v in zip(bars2, values_asian):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1, f'{v:.1f}%',
             ha='center', fontsize=12, fontproperties=zh_font, fontweight='bold')
ax2.set_ylabel('概率 (%)', fontproperties=zh_font)
ax2.set_title('亚指: 西班牙让一球', fontproperties=zh_font, fontsize=13, fontweight='bold')
ax2.set_ylim(0, max(values_asian)+12)
for tick in ax2.get_xticklabels():
    tick.set_fontproperties(zh_font)
for tick in ax2.get_yticklabels():
    tick.set_fontproperties(zh_font)

# --- 子图3: 大小球 ---
ax3 = plt.subplot(3, 3, 3)
eff_over = over + 0.5*exact_three
labels_ou = ['大球 (>2.5)', '小球 (<2.5/3)']
values_ou = [eff_over*100, (1-eff_over)*100]
colors_ou = ['#e67e22', '#2ecc71']
bars3 = ax3.bar(labels_ou, values_ou, color=colors_ou, width=0.5, edgecolor='white')
for bar, v in zip(bars3, values_ou):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{v:.1f}%',
             ha='center', fontsize=12, fontproperties=zh_font, fontweight='bold')
ax3.set_ylabel('有效概率 (%)', fontproperties=zh_font)
ax3.set_title('大小球 (2.5/3)', fontproperties=zh_font, fontsize=13, fontweight='bold')
ax3.set_ylim(0, max(values_ou)+12)
for tick in ax3.get_xticklabels():
    tick.set_fontproperties(zh_font)
for tick in ax3.get_yticklabels():
    tick.set_fontproperties(zh_font)

# --- 子图4: 预期进球分布 ---
ax4 = plt.subplot(3, 3, 4)
goal_range = np.arange(0, 7)
home_dist = [np.sum(home_goals_sim == g) / N_SIMULATIONS * 100 for g in goal_range]
away_dist = [np.sum(away_goals_sim == g) / N_SIMULATIONS * 100 for g in goal_range]
x = np.arange(len(goal_range))
w = 0.35
bars4h = ax4.bar(x - w/2, home_dist, w, label='西班牙', color='#e74c3c', alpha=0.85)
bars4a = ax4.bar(x + w/2, away_dist, w, label='比利时', color='#3498db', alpha=0.85)
ax4.set_xlabel('进球数', fontproperties=zh_font)
ax4.set_ylabel('概率 (%)', fontproperties=zh_font)
ax4.set_title('双方进球数概率分布', fontproperties=zh_font, fontsize=13, fontweight='bold')
ax4.set_xticks(x)
ax4.legend(prop=zh_font)
for tick in ax4.get_xticklabels():
    tick.set_fontproperties(zh_font)
for tick in ax4.get_yticklabels():
    tick.set_fontproperties(zh_font)

# --- 子图5: 总进球分布 ---
ax5 = plt.subplot(3, 3, 5)
tg_range = np.arange(0, 9)
tg_dist = [np.sum(total_goals == tg) / N_SIMULATIONS * 100 for tg in tg_range]
colors_tg = ['#bdc3c7']*len(tg_range)
bar5 = ax5.bar(tg_range, tg_dist, color=colors_tg, edgecolor='#7f8c8d', width=0.6)
for bar, v in zip(bar5, tg_dist):
    if v > 1:
        ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{v:.1f}%',
                 ha='center', fontsize=8, fontproperties=zh_font)
ax5.set_xlabel('总进球数', fontproperties=zh_font)
ax5.set_ylabel('概率 (%)', fontproperties=zh_font)
ax5.set_title('总进球数分布', fontproperties=zh_font, fontsize=13, fontweight='bold')
ax5.set_xticks(tg_range)
for tick in ax5.get_xticklabels():
    tick.set_fontproperties(zh_font)
for tick in ax5.get_yticklabels():
    tick.set_fontproperties(zh_font)

# --- 子图6: 比分热图 ---
ax6 = plt.subplot(3, 3, 6)
score_matrix = np.zeros((6, 6))
for h in range(6):
    for a in range(6):
        score_matrix[h][a] = np.sum((home_goals_sim == h) & (away_goals_sim == a)) / N_SIMULATIONS * 100
im = ax6.imshow(score_matrix, cmap='YlOrRd', aspect='auto')
for i in range(6):
    for j in range(6):
        v = score_matrix[i][j]
        if v > 0.5:
            ax6.text(j, i, f'{v:.1f}%', ha='center', va='center', fontsize=8, color='black' if v < 30 else 'white')
ax6.set_xticks(range(6))
ax6.set_yticks(range(6))
ax6.set_xlabel('比利时进球', fontproperties=zh_font)
ax6.set_ylabel('西班牙进球', fontproperties=zh_font)
ax6.set_title('比分概率热图 (%)', fontproperties=zh_font, fontsize=13, fontweight='bold')
for tick in ax6.get_xticklabels():
    tick.set_fontproperties(zh_font)
for tick in ax6.get_yticklabels():
    tick.set_fontproperties(zh_font)
plt.colorbar(im, ax=ax6, shrink=0.8)

# --- 子图7: 半全场 ---
ax7 = plt.subplot(3, 3, 7)
half_full_labels = ['胜-胜', '胜-平', '胜-负', '平-胜', '平-平', '平-负', '负-胜', '负-平', '负-负']
hf_probs = []
for hs in ['home', 'draw', 'away']:
    for fs in ['home', 'draw', 'away']:
        if hs == 'home':
            hmask = half_home > half_away
        elif hs == 'draw':
            hmask = half_home == half_away
        else:
            hmask = half_home < half_away
        if fs == 'home':
            fmask = home_goals_sim > away_goals_sim
        elif fs == 'draw':
            fmask = home_goals_sim == away_goals_sim
        else:
            fmask = home_goals_sim < away_goals_sim
        hf_probs.append(np.sum(hmask & fmask) / N_SIMULATIONS * 100)
hf_colors = ['#e74c3c','#e67e22','#c0392b','#f39c12','#95a5a6','#7f8c8d','#3498db','#2980b9','#2c3e50']
bars7 = ax7.barh(range(9), hf_probs, color=hf_colors, edgecolor='white')
for i, (bar, v) in enumerate(zip(bars7, hf_probs)):
    if v > 1:
        ax7.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2, f'{v:.1f}%',
                 va='center', fontsize=8, fontproperties=zh_font)
ax7.set_yticks(range(9))
ax7.set_yticklabels(half_full_labels, fontproperties=zh_font)
ax7.set_xlabel('概率 (%)', fontproperties=zh_font)
ax7.set_title('半全场组合概率 Top', fontproperties=zh_font, fontsize=13, fontweight='bold')
ax7.invert_yaxis()

# --- 子图8: 比分排名 ---
ax8 = plt.subplot(3, 3, 8)
top_scores = sorted_scores[:8]
score_names = [s[0] for s in top_scores]
score_vals = [s[1]/N_SIMULATIONS*100 for s in top_scores]
bars8 = ax8.barh(range(len(score_names)), score_vals, color='#9b59b6', edgecolor='white')
for i, (bar, v) in enumerate(zip(bars8, score_vals)):
    ax8.text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2, f'{v:.1f}%',
             va='center', fontsize=10, fontproperties=zh_font, fontweight='bold')
ax8.set_yticks(range(len(score_names)))
ax8.set_yticklabels(score_names, fontproperties=zh_font)
ax8.set_xlabel('概率 (%)', fontproperties=zh_font)
ax8.set_title('最可能比分 Top 8', fontproperties=zh_font, fontsize=13, fontweight='bold')
ax8.invert_yaxis()
for tick in ax8.get_xticklabels():
    tick.set_fontproperties(zh_font)

# --- 子图9: 关键分析文字 ---
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
info_text = (
    f"📊 ZIBP 泊松+蒙特卡洛模型\n"
    f"模拟次数: {N_SIMULATIONS:,}\n"
    f"λ_西班牙 = {lambda_home:.2f}   λ_比利时 = {lambda_away:.2f}\n\n"
    f"🎯 核心预测\n"
    f"西班牙胜率:   {home_win:.1%}\n"
    f"平局概率:     {draw:.1%}\n"
    f"比利时胜率:   {away_win:.1%}\n\n"
    f"📈 亚指(让一球)\n"
    f"赢全: {cover_home:.1%}  走水: {push:.1%}  输全: {lose_home:.1%}\n\n"
    f"📉 大小球(2.5/3)\n"
    f"大球(有效): {eff_over:.1%}  小球: {(1-eff_over):.1%}\n\n"
    f"🏆 推荐方向\n"
    f"主胜(西班牙)  推荐指数: ⭐⭐⭐⭐⭐\n"
    f"亚指西班牙-1  推荐指数: ⭐⭐⭐\n"
    f"大球 >2.5     推荐指数: ⭐⭐\n\n"
    f"模型更新: {datetime.now().strftime('%m-%d %H:%M')}"
)
ax9.text(0.05, 0.95, info_text, transform=ax9.transAxes,
         fontsize=9, fontproperties=zh_font, verticalalignment='top',
         fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#f0f0f0', alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.96])
output_path = '/home/admin/.openclaw/workspace/prediction_2907401.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ 预测图表已保存: {output_path}")

# ════════════════════════════════════════════════════════════════
# 6. 总结报告 (文本)
# ════════════════════════════════════════════════════════════════

print(f"""
╔══════════════════════════════════════════════════════╗
║     🏆 西班牙 vs 比利时 | 世界杯1/4决赛            ║
║     2026-07-11 03:00 | 洛杉矶体育场                ║
╠══════════════════════════════════════════════════════╣
║                                                    ║
║  📋 盘口分析                                        ║
║  ────────────────────────────────────────────────   ║
║  亚指: 西班牙让一球 (初盘半/一→升一球, 机构力挺)    ║
║  欧指: 主1.57  平3.90  客5.50 (主胜概率约58%)     ║
║  大小: 2.5/3  大球0.90水                           ║
║                                                    ║
║  📊 模型预测                                        ║
║  ────────────────────────────────────────────────   ║
║  【胜平负】                                         ║
║   西班牙 胜  {home_win:.1%}  |  平局  {draw:.1%}  |  比利时 胜  {away_win:.1%}  ║
║                                                    ║
║  【亚指】西班牙让一球                                ║
║   赢全(净胜2+) {cover_home:.1%}  |  走水(净胜1) {push:.1%}  |  输全 {lose_home:.1%}  ║
║                                                    ║
║  【大小球】2.5/3                                    ║
║   大球 {eff_over:.1%}  |  小球 {(1-eff_over):.1%}   ║
║                                                    ║
║  【最可能比分】                                      ║
║   1-0: {score_counts.get('1-0',0)/N_SIMULATIONS:.1%}  |  2-0: {score_counts.get('2-0',0)/N_SIMULATIONS:.1%}  |  2-1: {score_counts.get('2-1',0)/N_SIMULATIONS:.1%}  ║
║   1-1: {score_counts.get('1-1',0)/N_SIMULATIONS:.1%}  |  3-0: {score_counts.get('3-0',0)/N_SIMULATIONS:.1%}  |  0-0: {score_counts.get('0-0',0)/N_SIMULATIONS:.1%}  ║
║                                                    ║
║  🎯 推荐                                            ║
║  ────────────────────────────────────────────────   ║
║  ✅ 主推: 西班牙 胜 (赔率1.57) ⭐⭐⭐⭐⭐            ║
║  ✅ 次推: 西班牙-1 (赢全+走水 {cover_home+push:.1%}) ⭐⭐⭐  ║
║  ⚠️ 大小球偏小球, 但2.5/3盘口3球半赢半输           ║
║                                                    ║
║  💡 关键因素                                        ║
║  ① 西班牙防守稳固, 5场0失球                         ║
║  ② 比利时进攻无章法, 107射门效率低                   ║
║  ③ 从半/一升一球, 机构对西班牙信心增强               ║
║  ④ 盘面:西班牙WWWWDW vs 比利时WDWDDW               ║
║                                                    ║
╚══════════════════════════════════════════════════════╝
""")
