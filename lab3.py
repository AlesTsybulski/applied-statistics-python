"""
Лабораторная работа №3 — Вариант 32
Непараметрические критерии + описательная статистика
=====================================================
Задание 1  — Критерий знаков (Sign Test)
Задание 2  — Z-оценка
Задание 3  — T-критерий Уилкоксона (Wilcoxon Signed-Rank Test)
Задание 4  — U-критерий Манна–Уитни (Mann–Whitney U Test)
Задание 5  — H-критерий Краскела–Уоллиса (Kruskal–Wallis H Test)
Дополн.    — Коэффициент вариации (CV) для шкалы HADS
"""

import math

VARIANT = 32

# ─────────────────────────────────────────────────────────────────────────────
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────────────────────────────────────

def separator(title=""):
    line = "=" * 60
    if title:
        print(f"\n{line}")
        print(f"  {title}")
        print(line)
    else:
        print(line)


def mean(data):
    return sum(data) / len(data)


def sample_variance(data):
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / (len(data) - 1)


def sample_std(data):
    return math.sqrt(sample_variance(data))


def cv(data):
    """Коэффициент вариации (в %)."""
    m = mean(data)
    if m == 0:
        return float("inf")
    return abs(sample_std(data) / m) * 100


def assign_ranks(abs_values):
    """
    Присваивает ранги списку абсолютных значений.
    Возвращает список рангов (средние при совпадениях).
    Нули должны быть исключены ДО вызова функции.
    """
    n = len(abs_values)
    indexed = sorted(enumerate(abs_values), key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2  # средний ранг для группы
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


# ─────────────────────────────────────────────────────────────────────────────
#  ЗАДАНИЕ 1 — КРИТЕРИЙ ЗНАКОВ
# ─────────────────────────────────────────────────────────────────────────────

separator("ЗАДАНИЕ 1 — Критерий знаков (Sign Test)")

# Базовые данные из условия (шкала HADS, 0–21 баллов)
patients   = list(range(1, 13))
do_vals    = [18, 19,  4, 20, 17,  3, 18, 19,  5, 20, 16,  2]
posle_base = [12, 19,  5,  3, 16,  4, 18,  2,  6, 14, 15,  3]

# Правило для варианта: V >= 18 → вычесть V - 19 из всех «После»
shift = VARIANT - 19
posle_adj = [x - shift for x in posle_base]
print(f"Правило варианта {VARIANT}: V={VARIANT} >= 18 → сдвиг = {VARIANT} - 19 = {shift}")
print(f"Скорректированные значения «После»: {posle_adj}")

# Разности D = После − До
differences = [p - d for p, d in zip(posle_adj, do_vals)]

print(f"\n{'Пациент':>8} {'До':>5} {'После':>7} {'D=П−Д':>8} {'Знак':>6}")
print("-" * 40)
n_plus = n_minus = n_zero = 0
for i, (d_val, pos, diff) in enumerate(zip(do_vals, posle_adj, differences)):
    if diff > 0:
        sign = "+"
        n_plus += 1
    elif diff < 0:
        sign = "−"
        n_minus += 1
    else:
        sign = "0"
        n_zero += 1
    print(f"{patients[i]:>8} {d_val:>5} {pos:>7} {diff:>8} {sign:>6}")

n_eff = len(differences) - n_zero
S = min(n_plus, n_minus)
print(f"\nn+ = {n_plus}, n− = {n_minus}, нулевых = {n_zero}")
print(f"Эффективный n = {n_eff}")
print(f"S = min(n+, n−) = min({n_plus}, {n_minus}) = {S}")

# Критическое значение для n=12, α=0.05 (двусторонний)
S_cr = 2
alpha = 0.05
print(f"\nКритическое значение S_кр(n={n_eff}, α={alpha}) = {S_cr}")
if S <= S_cr:
    print(f"S = {S} ≤ S_кр = {S_cr}  →  H₀ ОТКЛОНЯЕТСЯ")
    print("Вывод: терапия статистически значимо снижает тревожность.")
else:
    print(f"S = {S} > S_кр = {S_cr}  →  H₀ НЕ отклоняется")


# ─────────────────────────────────────────────────────────────────────────────
#  ЗАДАНИЕ 2 — Z-ОЦЕНКА
# ─────────────────────────────────────────────────────────────────────────────

separator("ЗАДАНИЕ 2 — Z-оценка")

X   = 470   # нмоль/л — значение пациента (вариант 32)
mu  = 380   # нмоль/л — среднее популяции
sigma = 45  # нмоль/л — стандартное отклонение

print(f"X = {X} нмоль/л")
print(f"μ = {mu} нмоль/л")
print(f"σ = {sigma} нмоль/л")

numerator = X - mu
Z = numerator / sigma

print(f"\nПромежуточные вычисления:")
print(f"  X − μ = {X} − {mu} = {numerator}")
print(f"  Z = (X − μ) / σ = {numerator} / {sigma} = {Z:.4f}")

# P(Z < z) через функцию ошибок
p_less = 0.5 * (1 + math.erf(Z / math.sqrt(2)))
p_more = 1 - p_less
percentile = p_less * 100

print(f"\nZ = {Z:.2f}")
print(f"P(Z < {Z:.2f}) = {p_less:.4f}  →  {percentile:.2f}-й процентиль")
print(f"P(Z > {Z:.2f}) = {p_more:.4f}  ({p_more*100:.2f}% популяции имеют уровень выше)")
print("Вывод: кортизол пациента значительно выше нормы (2 СКО выше среднего).")


# ─────────────────────────────────────────────────────────────────────────────
#  ЗАДАНИЕ 3 — T-КРИТЕРИЙ УИЛКОКСОНА
# ─────────────────────────────────────────────────────────────────────────────

separator("ЗАДАНИЕ 3 — T-критерий Уилкоксона (Wilcoxon Signed-Rank Test)")

# Данные варианта 32
w_before = [18, 19,  4, 20, 17,  3, 18, 19,  5, 20, 16,  2]
w_after  = [29, 36, 22, 20, 33, 21, 35, 19, 23, 31, 32, 20]

print(f"{'Пациент':>8} {'До':>5} {'После':>7} {'D=П−Д':>8} {'|D|':>5} {'Знак':>6}")
print("-" * 45)

diffs_w = [a - b for a, b in zip(w_after, w_before)]
for i, (b, a, d) in enumerate(zip(w_before, w_after, diffs_w)):
    sign = "+" if d > 0 else ("−" if d < 0 else "0 (исключ.)")
    print(f"{i+1:>8} {b:>5} {a:>7} {d:>8} {abs(d):>5} {sign:>6}")

# Исключаем нулевые разности
non_zero = [(d, i+1) for i, d in enumerate(diffs_w) if d != 0]
excluded = [i+1 for i, d in enumerate(diffs_w) if d == 0]
n_w = len(non_zero)

print(f"\nИсключены пары (D=0): {excluded}")
print(f"Эффективный n = {n_w}")

# Ранжирование
abs_diffs = [abs(d) for d, _ in non_zero]
ranks_w   = assign_ranks(abs_diffs)
signs_w   = [1 if d > 0 else -1 for d, _ in non_zero]

print(f"\n{'Пациент':>8} {'|D|':>5} {'Ранг':>7} {'Знак':>6} {'Ранг со зн.':>12}")
print("-" * 45)
for (d, pat), r, s in zip(non_zero, ranks_w, signs_w):
    sign_str = "+" if s > 0 else "−"
    sr = f"+{r:.1f}" if s > 0 else f"−{r:.1f}"
    print(f"{pat:>8} {abs(d):>5} {r:>7.1f} {sign_str:>6} {sr:>12}")

W_plus  = sum(r for r, s in zip(ranks_w, signs_w) if s > 0)
W_minus = sum(r for r, s in zip(ranks_w, signs_w) if s < 0)
T_stat  = min(W_plus, W_minus)
control = n_w * (n_w + 1) / 2

print(f"\nW+ = {W_plus:.1f}")
print(f"W− = {W_minus:.1f}")
print(f"Контроль W+ + W− = {W_plus + W_minus:.1f}  (должно быть n(n+1)/2 = {control:.1f})")
print(f"T = min(W+, W−) = {T_stat:.1f}")

# Критическое значение для n=10, α=0.05
T_cr = 8
print(f"\nКритическое значение T_кр(n={n_w}, α=0.05) = {T_cr}")
if T_stat <= T_cr:
    print(f"T = {T_stat} ≤ T_кр = {T_cr}  →  H₀ ОТКЛОНЯЕТСЯ")
    print("Вывод: значимые изменения BDI-II после воздействия подтверждены.")
else:
    print(f"T = {T_stat} > T_кр = {T_cr}  →  H₀ НЕ отклоняется")


# ─────────────────────────────────────────────────────────────────────────────
#  ЗАДАНИЕ 4 — U-КРИТЕРИЙ МАННА–УИТНИ
# ─────────────────────────────────────────────────────────────────────────────

separator("ЗАДАНИЕ 4 — U-критерий Манна–Уитни (Mann–Whitney U Test)")

group_X = [7, 8, 6, 13, 5, 9, 10, 12]
group_Y = [17, 18, 16, 23, 15, 19, 20, 22]
n1, n2  = len(group_X), len(group_Y)
N_total = n1 + n2

print(f"Группа X (новая методика,  n₁={n1}): {group_X}")
print(f"Группа Y (традиц. методика, n₂={n2}): {group_Y}")

# Объединяем и сортируем
combined = [(v, "X") for v in group_X] + [(v, "Y") for v in group_Y]
combined.sort(key=lambda x: x[0])

# Присваиваем ранги (связанных нет в этом варианте)
ranked = []
i = 0
while i < len(combined):
    j = i
    while j < len(combined) - 1 and combined[j+1][0] == combined[i][0]:
        j += 1
    avg_r = (i + 1 + j + 1) / 2
    for k in range(i, j + 1):
        ranked.append((combined[k][0], combined[k][1], avg_r))
    i = j + 1

print(f"\n{'№':>4} {'Значение':>9} {'Группа':>7} {'Ранг':>6}")
print("-" * 30)
for idx, (val, grp, rnk) in enumerate(ranked, 1):
    print(f"{idx:>4} {val:>9} {grp:>7} {rnk:>6.1f}")

R1 = sum(r for v, g, r in ranked if g == "X")
R2 = sum(r for v, g, r in ranked if g == "Y")
control = N_total * (N_total + 1) / 2
print(f"\nR₁ (X) = {R1:.1f}")
print(f"R₂ (Y) = {R2:.1f}")
print(f"Контроль R₁+R₂ = {R1+R2:.1f}  (должно быть N(N+1)/2 = {control:.1f})")

U1 = n1 * n2 + n1 * (n1 + 1) / 2 - R1
U2 = n1 * n2 + n2 * (n2 + 1) / 2 - R2
U  = min(U1, U2)
print(f"\nU₁ = n₁·n₂ + n₁(n₁+1)/2 − R₁ = {n1}·{n2} + {n1*n2} + {n1*(n1+1)//2} − {R1:.0f} = {U1:.0f}")
print(f"U₂ = n₁·n₂ + n₂(n₂+1)/2 − R₂ = {n1}·{n2} + {n1*n2} + {n2*(n2+1)//2} − {R2:.0f} = {U2:.0f}")
print(f"Контроль U₁+U₂ = {U1+U2:.0f}  (должно быть n₁·n₂ = {n1*n2})")
print(f"U = min(U₁, U₂) = min({U1:.0f}, {U2:.0f}) = {U:.0f}")

# Критическое значение для n1=n2=8, α=0.05
U_cr = 13
print(f"\nКритическое значение U_кр(n₁={n1}, n₂={n2}, α=0.05) = {U_cr}")
if U <= U_cr:
    print(f"U = {U:.0f} ≤ U_кр = {U_cr}  →  H₀ ОТКЛОНЯЕТСЯ")
    print("Вывод: методики значимо различаются. Группа Y даёт более высокие результаты.")
else:
    print(f"U = {U:.0f} > U_кр = {U_cr}  →  H₀ НЕ отклоняется")


# ─────────────────────────────────────────────────────────────────────────────
#  ЗАДАНИЕ 5 — H-КРИТЕРИЙ КРАСКЕЛА–УОЛЛИСА
# ─────────────────────────────────────────────────────────────────────────────

separator("ЗАДАНИЕ 5 — H-критерий Краскела–Уоллиса (Kruskal–Wallis H Test)")

groups = {
    "A1": [73, 74, 75, 76, 78],
    "A2": [79, 80, 81, 82, 83],
    "A3": [71, 72, 73, 74, 75],
}
group_names = list(groups.keys())
group_sizes = {k: len(v) for k, v in groups.items()}
all_vals_kw = [(v, k) for k, vals in groups.items() for v in vals]
all_vals_kw.sort(key=lambda x: x[0])
N_kw = len(all_vals_kw)

for k, v in groups.items():
    print(f"  {k}: {v}  (n={len(v)})")
print(f"N = {N_kw}")

# Ранжирование с учётом связанных рангов
ranked_kw = []
i = 0
while i < N_kw:
    j = i
    while j < N_kw - 1 and all_vals_kw[j+1][0] == all_vals_kw[i][0]:
        j += 1
    avg_r = (i + 1 + j + 1) / 2
    for k_idx in range(i, j + 1):
        ranked_kw.append((all_vals_kw[k_idx][0], all_vals_kw[k_idx][1], avg_r))
    i = j + 1

print(f"\n{'Значение':>9} {'Группа':>7} {'Ранг':>7}")
print("-" * 28)
for val, grp, rnk in ranked_kw:
    print(f"{val:>9} {grp:>7} {rnk:>7.1f}")

# Суммы рангов
R = {k: 0.0 for k in group_names}
for val, grp, rnk in ranked_kw:
    R[grp] += rnk

control_kw = N_kw * (N_kw + 1) / 2
print(f"\nСуммы рангов:")
for k in group_names:
    print(f"  R({k}) = {R[k]:.1f}")
print(f"Контроль ΣRᵢ = {sum(R.values()):.1f}  (должно быть N(N+1)/2 = {control_kw:.1f})")

# H-статистика
sum_term = sum(R[k]**2 / group_sizes[k] for k in group_names)
H_raw = (12 / (N_kw * (N_kw + 1))) * sum_term - 3 * (N_kw + 1)

print(f"\nΣ(Rᵢ²/nᵢ):")
for k in group_names:
    print(f"  {k}: {R[k]:.1f}² / {group_sizes[k]} = {R[k]**2:.2f} / {group_sizes[k]} = {R[k]**2/group_sizes[k]:.4f}")
print(f"  Итого = {sum_term:.4f}")
print(f"\nH = [12 / ({N_kw}·{N_kw+1})] · {sum_term:.4f} − 3·{N_kw+1}")
print(f"H = {12/(N_kw*(N_kw+1)):.6f} · {sum_term:.4f} − {3*(N_kw+1)}")
print(f"H = {12/(N_kw*(N_kw+1))*sum_term:.4f} − {3*(N_kw+1)}")
print(f"H = {H_raw:.4f}")

# Поправка на связанные ранги
tie_correction = 0.0
# Считаем группы повторений
from collections import Counter
val_counts = Counter(v for v, _ in all_vals_kw)
ties_info = {v: c for v, c in val_counts.items() if c > 1}
print(f"\nСвязанные ранги: {ties_info}")
sum_T = sum(t**3 - t for t in ties_info.values())
print(f"ΣT = Σ(tᵢ³ − tᵢ) = {' + '.join(str(t**3-t) for t in ties_info.values())} = {sum_T}")
C = 1 - sum_T / (N_kw**3 - N_kw)
print(f"C = 1 − {sum_T} / ({N_kw}³ − {N_kw}) = 1 − {sum_T} / {N_kw**3 - N_kw} = {C:.6f}")
H_corr = H_raw / C
print(f"H_скорр = {H_raw:.4f} / {C:.6f} = {H_corr:.4f}")

# Критическое значение χ²(df=2, α=0.05)
chi2_cr = 5.991
df = len(group_names) - 1
print(f"\ndf = k − 1 = {len(group_names)} − 1 = {df}")
print(f"χ²_кр(df={df}, α=0.05) = {chi2_cr}")
if H_corr > chi2_cr:
    print(f"H_скорр = {H_corr:.2f} > χ²_кр = {chi2_cr}  →  H₀ ОТКЛОНЯЕТСЯ")
    print("Вывод: между группами есть значимые различия (A2 > A1 > A3).")
else:
    print(f"H_скорр = {H_corr:.2f} ≤ χ²_кр = {chi2_cr}  →  H₀ НЕ отклоняется")


# ─────────────────────────────────────────────────────────────────────────────
#  ДОПОЛНИТЕЛЬНО — КОЭФФИЦИЕНТ ВАРИАЦИИ (ШКАЛА HADS)
# ─────────────────────────────────────────────────────────────────────────────

separator("ДОПОЛНИТЕЛЬНО — Коэффициент вариации (шкала HADS)")

for label, data in [("До терапии", do_vals), ("После терапии (скорр.)", posle_adj)]:
    n   = len(data)
    s   = sum(data)
    m   = mean(data)
    devs = [x - m for x in data]
    sq   = [d**2 for d in devs]
    ssq  = sum(sq)
    var  = ssq / (n - 1)
    std  = math.sqrt(var)
    cv_v = abs(std / m) * 100 if m != 0 else float("inf")

    print(f"\n--- {label} ---")
    print(f"Данные: {data}")
    print(f"n = {n},  Σxᵢ = {s},  x̄ = {s}/{n} = {m:.4f}")
    print(f"\n{'i':>3} {'xᵢ':>5} {'xᵢ−x̄':>9} {'(xᵢ−x̄)²':>12}")
    print("-" * 33)
    for i, (xi, d, sq_d) in enumerate(zip(data, devs, sq), 1):
        print(f"{i:>3} {xi:>5} {d:>9.4f} {sq_d:>12.4f}")
    print("-" * 33)
    print(f"{'Σ':>3} {s:>5} {'—':>9} {ssq:>12.4f}")
    print(f"\ns² = Σ(xᵢ−x̄)² / (n−1) = {ssq:.4f} / {n-1} = {var:.4f}")
    print(f"σ  = √{var:.4f} = {std:.4f}")
    print(f"CV = σ / |x̄| × 100% = {std:.4f} / |{m:.4f}| × 100% = {cv_v:.2f}%")
    if abs(m) < 5:
        print("  ⚠  |x̄| мало — CV не информативен, используйте σ напрямую.")
    elif cv_v > 33:
        print("  → Высокая неоднородность данных (CV > 33%).")
    else:
        print("  → Умеренная неоднородность данных (CV ≤ 33%).")


# ─────────────────────────────────────────────────────────────────────────────
#  ИТОГОВАЯ СВОДКА
# ─────────────────────────────────────────────────────────────────────────────

separator("ИТОГОВАЯ СВОДКА — Вариант 32")
print(f"  Задание 1 (Критерий знаков):   S = 0 ≤ S_кр = 2   → H₀ отклоняется")
print(f"  Задание 2 (Z-оценка):           Z = 2.00  → 97.72-й процентиль")
print(f"  Задание 3 (Уилкоксон):          T = 0 ≤ T_кр = 8   → H₀ отклоняется")
print(f"  Задание 4 (Манна–Уитни):        U = 0 ≤ U_кр = 13  → H₀ отклоняется")
print(f"  Задание 5 (Краскел–Уоллис):     H = {H_corr:.2f} > χ²_кр = 5.991 → H₀ отклоняется")
print(f"  CV (HADS До):                   CV = {cv(do_vals):.2f}%  — высокая неоднородность")
print(f"  CV (HADS После):                σ  = {sample_std(posle_adj):.2f} (CV не информативен)")
separator()
