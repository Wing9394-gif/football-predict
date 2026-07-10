#!/usr/bin/env python3
"""
ZIBP Poisson + Monte Carlo prediction for Spain vs Belgium
Match ID: 2907401 | 2026 World Cup Quarter-Final
"""
import math
import random
import json
from collections import Counter

random.seed(42)  # deterministic

# ============================================================
# MATCH DATA (from nowscore.com - 2026-07-10 22:54 CST)
# ============================================================
match = {
    "id": 2907401,
    "home": "西班牙",
    "away": "比利时",
    "league": "世界杯",
    "time": "2026-07-11 03:00 CST",
    "venue": "洛杉矶体育场",
    "weather": "天晴 24~25°C",
}

# ---- League Profile (World Cup) ----
lp = {
    "avg_goals": 2.55,
    "over25_rate": 0.48,
    "zero_inflated_pi": 0.08,
    "home_advantage": 0.20,  # neutral venue
    "lambda_range": [1.0, 1.8],
    "time_decay_lambda": 0.04,
    "fatigue_sensitivity": "high",
    "capital_flow_credibility": "high",
}

# ---- Team Stats (from analysis page + Sky Sports info) ----
# Spain: 5 matches, 4W 1D 0L, 0 goals conceded, WWWWDW form
# avg goals scored: 1.8/home 2.33, avg conceded: 0
# Key: Oyarzabal 4 goals, Yamal creator, Cubarsi/Laporte solid defense

# Belgium: 5 matches, 3W 1D 1L (to ET/pen), form WDWDDW
# avg goals scored: 2.4/overall, 4.5/away
# Key: De Ketelaere 2g vs USA, Trossard 2g2a, Lukaku 3g

spain = {
    "name": "西班牙",
    "recent_home_goals": 2.33,  # avg goals scored (home/nuetral)
    "recent_home_conceded": 0.0,
    "h2h_home_goals": 1.5,     # estimated from similar opponents
    "h2h_home_conceded": 0.8,
    "xg_for": 1.9,             # tournament xG average (from Sky Sports: underperforming but creating)
    "xga_against": 0.9,        # low xGA of just 1.49 total in 5 matches
    "recent_avg_rating": 6.85, # from nowscore
    "last_5_ratings": [6.85, 7.53, 6.91, 7.22, 6.81],
    "key_miss": [],
}

belgium = {
    "name": "比利时",
    "recent_away_goals": 4.5,
    "recent_away_conceded": 1.0,
    "h2h_away_goals": 1.2,
    "h2h_away_conceded": 1.8,
    "xg_for": 1.7,
    "xga_against": 1.5,
    "recent_avg_rating": 7.04,
    "last_5_ratings": [7.04, 6.96, 7.12, 6.65, 6.62],
    "key_miss": ["Amadou Onana (midfield, ACL)", "Zeno Debast (defender, illness)"],
}

# ---- Odds (Crown* / 皇冠即时赔率 - current) ----
odds = {
    "asian": {
        "line": 1.0,  # 一球
        "home_water": 1.03,
        "away_water": 0.86,
        "early_line": 0.75,  # 半/一
        "early_home": 0.80,
        "early_away": 1.02,
    },
    "euro": {
        "home": 1.60,
        "draw": 4.15,
        "away": 4.90,
        "early_home": 1.65,
        "early_draw": 4.15,
        "early_away": 4.90,
    },
    "ou": {
        "line": 2.5,
        "over_water": 1.00,
        "under_water": 0.88,
        "early_line": 2.5,
        "early_over": 0.80,
        "early_under": 1.00,
    },
}

# ---- Referee: Michael Oliver (ENG) ----
ref = {
    "name": "Michael Oliver",
    "total_matches": 536,
    "avg_fouls": 11.12,
    "avg_yellow": 1.71,
    "avg_red": 0.08,
    "home_win_rate": 0.40,
    "away_win_rate": 0.38,
    "ref_spain_record": "5场(3胜2平0负), 场均犯规10.4, 黄1.4, 红0",
    "ref_belgium_record": "1场(1胜0平0负), 场均犯规8, 黄3, 红0",
}

# ============================================================
# STEP 1: ZIBP Model - Lambda estimation
# ============================================================
def compute_lambda(home_goals, h2h_goals, xg, weight_h=0.5, weight_h2h=0.2, weight_xg=0.3):
    return home_goals * weight_h + h2h_goals * weight_h2h + xg * weight_xg

lambda_home = compute_lambda(
    spain["recent_home_goals"],
    spain["h2h_home_goals"],
    spain["xg_for"]
)
lambda_away = compute_lambda(
    belgium["recent_away_goals"],
    belgium["h2h_away_goals"],
    belgium["xga_against"]  # using xGA for away team's expected goals against
)

# Time decay factor (last match was ~3 days ago for both - quarter-finals are packed)
days_since_last = 3
time_decay = math.exp(-lp["time_decay_lambda"] * days_since_last)
lambda_home *= time_decay
lambda_away *= time_decay

# Fatigue factor (World Cup knockout - high density)
fatigue_factor = 1 + (days_since_last / 7 - 1) * 0.08  # δ=0.08 for World Cup KO
# Actually let's compute more precisely: ≤3d => 0.85~0.92
# Spain played Portugal on July 6 (4 days ago), Belgium played USA on July 5 (5 days ago)
# But from Sky: quarter-finals "begin today" = July 9, so Spain's last-16 was about 3 days before
fatigue_adj = 0.90  # ~3-4 day gap

lambda_home_final = max(lp["lambda_range"][0], min(lambda_home * fatigue_adj, lp["lambda_range"][1]))
lambda_away_final = max(lp["lambda_range"][0], min(lambda_away * fatigue_adj, lp["lambda_range"][1]))

# ============================================================
# STEP 2: Monte Carlo Simulation (10,000 runs)
# ============================================================
pi = lp["zero_inflated_pi"]  # zero-inflation parameter

def poisson_prob(k, lmbda):
    """Poisson PMF"""
    return (lmbda ** k) * math.exp(-lmbda) / math.factorial(k)

def biv_poisson_prob(h, a, l1, l2):
    """Independent bivariate Poisson"""
    return poisson_prob(h, l1) * poisson_prob(a, l2)

def zibp_prob(h, a, l1, l2, pi):
    """Zero-inflated bivariate Poisson"""
    if h == 0 and a == 0:
        return pi + (1 - pi) * biv_poisson_prob(0, 0, l1, l2)
    else:
        return (1 - pi) * biv_poisson_prob(h, a, l1, l2)

N_SIM = 10000
MAX_GOALS = 10

results = Counter()
probs = {}
total_prob = 0

for h in range(MAX_GOALS + 1):
    for a in range(MAX_GOALS + 1):
        p = zibp_prob(h, a, lambda_home_final, lambda_away_final, pi)
        probs[(h, a)] = p
        total_prob += p

# Normalize
for k in probs:
    probs[k] /= total_prob

# Monte Carlo simulation
for _ in range(N_SIM):
    r = random.random()
    cum = 0
    for (h, a), p in sorted(probs.items(), key=lambda x: -x[1]):
        cum += p
        if r <= cum:
            results[(h, a)] += 1
            break

# Aggregate
mc = {k: v / N_SIM for k, v in results.items()}
top10 = sorted(mc.items(), key=lambda x: -x[1])[:10]

# Match result probs
home_win = sum(p for (h, a), p in mc.items() if h > a)
draw = sum(p for (h, a), p in mc.items() if h == a)
away_win = sum(p for (h, a), p in mc.items() if a > h)

# Over/Under probs
over_25 = sum(p for (h, a), p in mc.items() if h + a >= 3)
under_25 = sum(p for (h, a), p in mc.items() if h + a <= 2)
over_15 = sum(p for (h, a), p in mc.items() if h + a >= 2)
under_15 = sum(p for (h, a), p in mc.items() if h + a <= 1)
over_35 = sum(p for (h, a), p in mc.items() if h + a >= 4)
under_35 = sum(p for (h, a), p in mc.items() if h + a <= 3)

# Asian handicap: Spain -1
# Win by 2+ = home -1 W, Win by 1 = push, Draw/loss = home -1 L
asian_1_win = sum(p for (h, a), p in mc.items() if h - a >= 2)
asian_1_push = sum(p for (h, a), p in mc.items() if h - a == 1)
asian_1_loss = sum(p for (h, a), p in mc.items() if h - a <= 0)

# Asian handicap 0.75 (half push)
asian_075_win = sum(p for (h, a), p in mc.items() if h - a >= 2)
asian_075_half = sum(p for (h, a), p in mc.items() if h - a == 1)
asian_075_loss = sum(p for (h, a), p in mc.items() if h - a <= 0)
# For -0.75: win by 2+ = full, win by 1 = half, draw/loss = loss
asian_075_full_win = asian_1_win
asian_075_half_win = asian_1_push
asian_075_full_loss = asian_1_loss

# ============================================================
# STEP 3: Expected Value
# ============================================================
# Crown* odds
ev_home = home_win * odds["euro"]["home"] - (1 - home_win)
ev_draw = draw * odds["euro"]["draw"] - (1 - draw)
ev_away = away_win * odds["euro"]["away"] - (1 - away_win)

# Asian -1 (一球) at 1.03 water
ev_asian_home = asian_1_win * (1 + 1.03 - 1) + asian_1_push * 1.0 - asian_1_loss * 1.0
# scaled: win full, push push, loss full
total_asian_exp = asian_1_win * 1.03 + asian_1_push * 0 - asian_1_loss * 1.0

# Over 2.5
ev_over25 = over_25 * odds["ou"]["over_water"] - (1 - over_25)
ev_under25 = under_25 * odds["ou"]["under_water"] - (1 - under_25)

# ============================================================
# STEP 4: Detect anomalies / mispricing
# ============================================================
# Early vs current: Asian line changed from 0.75 to 1.0 → significant move
asian_line_shift = odds["asian"]["line"] - odds["asian"]["early_line"]
euro_home_shift = odds["euro"]["early_home"] - odds["euro"]["home"]

# AI theoretical line (based on home_win prob)
# home_win 0.598 → implied line ~0.75 to 1.0 range
implied_line = home_win * 2 - 0.5  # rough estimate
actual_line = odds["asian"]["line"]
mispricing = implied_line - actual_line

# ============================================================
# STEP 5: Dutching Best Correct Scores
# ============================================================
# Get top CS probabilities
top_cs = [(f"{h}-{a}", p, float(h), float(a)) for (h, a), p in top10[:10]]
# Sort by probability descending

# Crown* typical CS odds for this match (estimated from Euro odds)
# Home 1.60, Draw 4.15, Away 4.90 → most CS odds in range
cs_odds_est = {
    "1-0": 6.5, "2-0": 7.5, "2-1": 8.0, "3-0": 12.0, "3-1": 13.0,
    "1-1": 7.0, "0-0": 10.0, "2-2": 15.0,
    "0-1": 12.0, "1-2": 14.0, "0-2": 18.0,
}

BANKROLL = 100

def dutching(scores, odds_dict):
    """Compute Dutching allocation"""
    if not scores:
        return None
    inv_sum = sum(1.0 / odds_dict.get(s, 50) for s in scores if odds_dict.get(s, 50) > 0)
    if inv_sum <= 0:
        return None
    return_rate = 1.0 / inv_sum
    stakes = {}
    for s in scores:
        o = odds_dict.get(s, 50)
        if o > 0:
            stakes[s] = (1.0 / o) / inv_sum * BANKROLL
    return stakes, return_rate

# Pick best 3-5 scores for dutching (diversified across outcomes)
# Goal: cover multiple scenarios with limited total stake
dutch_candidates = ["1-0", "2-0", "1-1", "2-1", "3-0"]
dutch_result = dutching(dutch_candidates, cs_odds_est)
dutch_prob = sum(probs.get((int(s.split("-")[0]), int(s.split("-")[1])), 0) for s in dutch_candidates)

# ============================================================
# OUTPUT
# ============================================================
output = {
    "match": match,
    "lambda_home": round(lambda_home_final, 3),
    "lambda_away": round(lambda_away_final, 3),
    "pi": round(pi, 3),
    "result_probs": {
        "home_win": round(home_win * 100, 1),
        "draw": round(draw * 100, 1),
        "away_win": round(away_win * 100, 1),
    },
    "top10_scores": [
        {"score": f"{h}-{a}", "prob": round(p * 100, 2)}
        for (h, a), p in top10
    ],
    "asian_handicap": {
        "line": "-1",
        "home_water": 1.03,
        "win_by_2plus": round(asian_1_win * 100, 1),
        "win_by_1": round(asian_1_push * 100, 1),
        "draw_or_loss": round(asian_1_loss * 100, 1),
        "ev": round(ev_asian_home, 3),
    },
    "asian_075": {
        "line": "-0.75",
        "full_win": round(asian_075_full_win * 100, 1),
        "half_win": round(asian_075_half_win * 100, 1),
        "full_loss": round(asian_075_full_loss * 100, 1),
    },
    "over_under": {
        "over_25_prob": round(over_25 * 100, 1),
        "under_25_prob": round(under_25 * 100, 1),
        "over_25_ev": round(ev_over25, 3),
        "under_25_ev": round(ev_under25, 3),
        "over_15_prob": round(over_15 * 100, 1),
        "under_15_prob": round(under_15 * 100, 1),
    },
    "euro": {
        "home_ev": round(ev_home, 3),
        "draw_ev": round(ev_draw, 3),
        "away_ev": round(ev_away, 3),
        "fair_home": round(1/home_win, 3),
        "fair_draw": round(1/draw, 3),
        "fair_away": round(1/away_win, 3),
    },
    "mispricing": {
        "asian_line_shift": asian_line_shift,
        "implied_fair_line": round(implied_line, 2),
        "actual_line": actual_line,
        "mispricing_delta": round(mispricing, 2),
        "euro_home_shift": round(euro_home_shift, 2),
    },
    "dutching": {
        "scores": dutch_candidates,
        "allocations": {s: round(dutch_result[0][s], 1) for s in dutch_candidates} if dutch_result else {},
        "total_stake": BANKROLL,
        "return_per_score": {s: round(dutch_result[0][s] * cs_odds_est.get(s, 1), 1) if dutch_result else 0 for s in dutch_candidates},
        "return_rate": round(dutch_result[1], 3) if dutch_result else 0,
        "coverage_prob": round(dutch_prob * 100, 1),
    },
    "monte_carlo_runs": N_SIM,
    "rating": {
        "home_avg": round(spain["recent_avg_rating"], 2),
        "away_avg": round(belgium["recent_avg_rating"], 2),
    },
    "anomaly_flags": [],
}

# Check anomalies
if mispricing >= 0.25:
    output["anomaly_flags"].append(f"盘口偏差：理论{implied_line:.2f} vs 实际{actual_line:.0f}，偏差{mispricing:.2f}球")
if asian_line_shift >= 0.25:
    output["anomaly_flags"].append(f"盘口升幅：从{odds['asian']['early_line']}升到{odds['asian']['line']}（{asian_line_shift:.2f}球），庄家看好西班牙")
if spain["xg_for"] - belgium["xga_against"] > 0.5:
    output["anomaly_flags"].append("xG差距显著：西班牙xG创造优势明显")
if belgium["key_miss"]:
    output["anomaly_flags"].append(f"比利时伤缺：{'/'.join(belgium['key_miss'])}")

# Print JSON
print(json.dumps(output, ensure_ascii=False, indent=2))
