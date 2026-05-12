import math
from scipy import stats

# ============================================================
# ЛАБОРАТОРНАЯ РАБОТА №5 — ВАРИАНТ 32
# ============================================================

# ============================================================
# ЗАДАНИЕ 1 — КОЭФФИЦИЕНТ КОРРЕЛЯЦИИ ПИРСОНА
# ============================================================

data1 = [
    (5,85),(10,80),(15,75),(20,70),(25,65),(30,60),(35,55),(40,50),(45,46),(50,43),
    (4,87),(9,82),(14,77),(19,72),(24,67),(29,62),(34,57),(39,52),(44,48),(49,45),
    (3,89),(8,84),(13,79),(18,74),(23,69),(28,64),(33,59),(38,54),(43,49),(48,46),
    (6,83),(11,78),(16,73),(21,68),(26,63),(31,58),(36,53),(41,48)
]

X = [d[0] for d in data1]
Y = [d[1] for d in data1]
n = len(X)

# Вспомогательные суммы
sumX  = sum(X)
sumY  = sum(Y)
sumXY = sum(x*y for x, y in zip(X, Y))
sumX2 = sum(x**2 for x in X)
sumY2 = sum(y**2 for y in Y)
meanX = sumX / n
meanY = sumY / n

# Коэффициент корреляции Пирсона
numerator   = n * sumXY - sumX * sumY
den_left    = n * sumX2 - sumX**2
den_right   = n * sumY2 - sumY**2
denominator = math.sqrt(den_left * den_right)
r  = numerator / denominator
R2 = r**2

# t-критерий Стьюдента
df     = n - 2
t_stat = r * math.sqrt(df) / math.sqrt(1 - r**2)
t_crit = stats.t.ppf(0.975, df=df)          # двусторонний, α=0.05
p_val  = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))

print("=" * 50)
print("ЗАДАНИЕ 1 — КОЭФФИЦИЕНТ ПИРСОНА")
print("=" * 50)
print(f"n          = {n}")
print(f"ΣX         = {sumX}")
print(f"ΣY         = {sumY}")
print(f"ΣXY        = {sumXY}")
print(f"ΣX²        = {sumX2}")
print(f"ΣY²        = {sumY2}")
print(f"x̄          = {meanX:.4f}")
print(f"ȳ          = {meanY:.4f}")
print(f"Числитель  = {numerator}")
print(f"Левая ч.   = {den_left}")
print(f"Правая ч.  = {den_right}")
print(f"Знаменат.  = {denominator:.4f}")
print(f"r          = {r:.4f}")
print(f"R²         = {R2:.4f}")
print(f"t_набл     = {t_stat:.4f}")
print(f"df         = {df}")
print(f"t_крит     = {t_crit:.4f}  (α=0.05, двусторонний)")
print(f"p-value    = {p_val:.6f}")
print()
if abs(t_stat) > t_crit:
    print(f"|t_набл| = {abs(t_stat):.4f} > t_крит = {t_crit:.4f}")
    print("→ Корреляция СТАТИСТИЧЕСКИ ЗНАЧИМА, H₀ отвергается")
else:
    print(f"|t_набл| = {abs(t_stat):.4f} < t_крит = {t_crit:.4f}")
    print("→ Корреляция НЕ значима, H₀ не отвергается")

direction = "положительная (прямая)" if r > 0 else "отрицательная (обратная)"

if abs(r) < 0.20:
    strength = "очень слабая"
elif abs(r) < 0.40:
    strength = "слабая"
elif abs(r) < 0.60:
    strength = "умеренная"
elif abs(r) < 0.80:
    strength = "сильная"
else:
    strength = "очень сильная"

print(f"Направление: {direction}")
print(f"Сила связи:  {strength}")


# ============================================================
# ЗАДАНИЕ 2 — КОЭФФИЦИЕНТ РАНГОВОЙ КОРРЕЛЯЦИИ СПИРМЕНА
# ============================================================

data2 = [
    (12,52),(15,50),(28,60),(22,61),(24,58),
    (13,50),(19,55),(15,53),(20,55),(19,54),
    (13,51),(23,59),(16,53),(18,53),(24,57)
]

X2 = [d[0] for d in data2]
Y2 = [d[1] for d in data2]
n2 = len(X2)

def rank_with_ties(arr):
    """Ранжирование с учётом связанных рангов (средний ранг при совпадениях)"""
    sorted_vals = sorted(arr)
    ranks = []
    for x in arr:
        positions = [i + 1 for i, v in enumerate(sorted_vals) if v == x]
        ranks.append(sum(positions) / len(positions))
    return ranks

Rx    = rank_with_ties(X2)
Ry    = rank_with_ties(Y2)
d     = [Rx[i] - Ry[i] for i in range(n2)]
d2    = [di**2 for di in d]
sumD2 = sum(d2)

rs      = 1 - 6 * sumD2 / (n2 * (n2**2 - 1))
df2     = n2 - 2
t_s     = rs * math.sqrt(df2) / math.sqrt(1 - rs**2)
t_crit_s = stats.t.ppf(0.975, df=df2)       # двусторонний, α=0.05
p_val_s  = 2 * (1 - stats.t.cdf(abs(t_s), df=df2))

print()
print("=" * 50)
print("ЗАДАНИЕ 2 — КОЭФФИЦИЕНТ СПИРМЕНА")
print("=" * 50)
print(f"{'№':>3} {'X':>4} {'Rx':>5} {'Y':>4} {'Ry':>5} {'d':>6} {'d²':>6}")
print("-" * 38)
for i in range(n2):
    sign = "+" if d[i] >= 0 else ""
    print(f"{i+1:>3} {X2[i]:>4} {Rx[i]:>5.1f} {Y2[i]:>4} {Ry[i]:>5.1f} "
          f"{sign}{d[i]:>5.1f} {d2[i]:>6.2f}")
print("-" * 38)
print(f"{'Σd²':>33} {sumD2:>6.2f}")
print()
print(f"n          = {n2}")
print(f"Σd²        = {sumD2:.2f}")
print(f"rs         = 1 - 6·{sumD2} / [{n2}·({n2**2}-1)]")
print(f"rs         = 1 - {6*sumD2} / {n2*(n2**2-1)}")
print(f"rs         = 1 - {6*sumD2 / (n2*(n2**2-1)):.4f}")
print(f"rs         = {rs:.4f}")
print(f"t_набл     = {t_s:.4f}")
print(f"df         = {df2}")
print(f"t_крит     = {t_crit_s:.4f}  (α=0.05, двусторонний)")
print(f"p-value    = {p_val_s:.6f}")
print()
if abs(t_s) > t_crit_s:
    print(f"|t_набл| = {abs(t_s):.4f} > t_крит = {t_crit_s:.4f}")
    print("→ Корреляция СТАТИСТИЧЕСКИ ЗНАЧИМА, H₀ отвергается")
else:
    print(f"|t_набл| = {abs(t_s):.4f} < t_крит = {t_crit_s:.4f}")
    print("→ Корреляция НЕ значима, H₀ не отвергается")

direction_s = "положительная (прямая)" if rs > 0 else "отрицательная (обратная)"

if abs(rs) < 0.20:
    strength_s = "очень слабая"
elif abs(rs) < 0.40:
    strength_s = "слабая"
elif abs(rs) < 0.60:
    strength_s = "умеренная"
elif abs(rs) < 0.80:
    strength_s = "сильная"
else:
    strength_s = "очень сильная"

print(f"Направление: {direction_s}")
print(f"Сила связи:  {strength_s}")