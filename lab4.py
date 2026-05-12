"""
Лабораторная работа №4 — Дисперсионный анализ (ANOVA)
Вариант 32
"""

import numpy as np
from scipy import stats


# ══════════════════════════════════════════════════════════════
#  ЗАДАНИЕ 1 — Однофакторный ANOVA (независимые выборки)
# ══════════════════════════════════════════════════════════════

print("=" * 60)
print("ЗАДАНИЕ 1. Однофакторный ANOVA (независимые выборки)")
print("Влияние возраста на длительность госпитализации")
print("=" * 60)

# Данные вариант 32
g1 = [2, 2, 2, 3]   # Группа 1 — младше 45 лет
g2 = [5, 5, 6, 6]   # Группа 2 — 45–55 лет
g3 = [11, 11, 12, 12]  # Группа 3 — старше 55 лет

groups = [g1, g2, g3]
n = [len(g) for g in groups]   # размеры групп
N = sum(n)                      # общий объём выборки
k = len(groups)                 # количество групп

# Групповые и общее средние
means = [np.mean(g) for g in groups]
grand_mean = np.mean([x for g in groups for x in g])

print("\nИсходные данные:")
for i, (g, m) in enumerate(zip(groups, means), 1):
    print(f"  Группа {i}: {g}  →  n={len(g)}, x̄={m:.2f}")
print(f"  Общее среднее x̄ = {grand_mean:.4f}")

# --- Суммы квадратов ---
SSB = sum(ni * (mi - grand_mean) ** 2 for ni, mi in zip(n, means))
SSW = sum((x - mi) ** 2 for g, mi in zip(groups, means) for x in g)
SST = SSB + SSW

dfB = k - 1
dfW = N - k
dfT = N - 1

MSB = SSB / dfB
MSW = SSW / dfW

F_obs = MSB / MSW
F_crit = stats.f.ppf(0.95, dfB, dfW)
p_value = 1 - stats.f.cdf(F_obs, dfB, dfW)

print("\nТаблица ANOVA:")
print(f"{'Источник':<25} {'SS':>10} {'df':>5} {'MS':>10} {'F набл.':>10} {'F крит.':>10}")
print("-" * 72)
print(f"{'Между группами':<25} {SSB:>10.4f} {dfB:>5} {MSB:>10.4f} {F_obs:>10.4f} {F_crit:>10.4f}")
print(f"{'Внутри групп':<25} {SSW:>10.4f} {dfW:>5} {MSW:>10.4f} {'—':>10} {'—':>10}")
print(f"{'Итого':<25} {SST:>10.4f} {dfT:>5} {'—':>10} {'—':>10} {'—':>10}")

print(f"\np-value = {p_value:.2e}")
if p_value < 0.05:
    print("Вывод: H₀ ОТВЕРГАЕТСЯ — возраст значимо влияет на длительность")
    print("       госпитализации (p < 0,05).")
else:
    print("Вывод: H₀ не отвергается — значимого влияния не обнаружено.")


# ══════════════════════════════════════════════════════════════
#  ЗАДАНИЕ 2 — Repeated Measures ANOVA (связанные выборки)
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ЗАДАНИЕ 2. Repeated Measures ANOVA (связанные выборки)")
print("Влияние времени лечения на систолическое АД")
print("=" * 60)

# Данные вариант 32: каждая строка — один пациент (A, B, C)
data = np.array([
    [134, 130, 126],
    [136, 128, 124],
    [133, 131, 127],
    [135, 129, 125],
    [137, 132, 128],
    [132, 127, 123],
])

n_subj, k2 = data.shape   # 6 пациентов, 3 условия
N2 = n_subj * k2

cond_means = data.mean(axis=0)   # средние по условиям (столбцам)
subj_means = data.mean(axis=1)   # средние по субъектам (строкам)
grand_mean2 = data.mean()

print("\nДанные (строки = пациенты, столбцы = A / B / C):")
print(f"{'Пациент':>10} {'A':>8} {'B':>8} {'C':>8} {'Среднее':>10}")
print("-" * 46)
for i, (row, sm) in enumerate(zip(data, subj_means), 1):
    print(f"{i:>10} {row[0]:>8} {row[1]:>8} {row[2]:>8} {sm:>10.2f}")
print("-" * 46)
print(f"{'Среднее':>10} {cond_means[0]:>8.2f} {cond_means[1]:>8.2f} {cond_means[2]:>8.2f} {grand_mean2:>10.2f}")

# --- Суммы квадратов ---
SS_total = np.sum((data - grand_mean2) ** 2)
SS_cond  = n_subj * np.sum((cond_means - grand_mean2) ** 2)
SS_subj  = k2     * np.sum((subj_means - grand_mean2) ** 2)
SS_error = SS_total - SS_cond - SS_subj

df_cond  = k2 - 1
df_subj  = n_subj - 1
df_error = df_cond * df_subj
df_total = N2 - 1

MS_cond  = SS_cond  / df_cond
MS_error = SS_error / df_error

F2_obs  = MS_cond / MS_error
F2_crit = stats.f.ppf(0.95, df_cond, df_error)
p2_value = 1 - stats.f.cdf(F2_obs, df_cond, df_error)

print("\nТаблица ANOVA:")
print(f"{'Источник':<25} {'SS':>10} {'df':>5} {'MS':>10} {'F набл.':>10} {'F крит.':>10}")
print("-" * 72)
print(f"{'Условия (лечение)':<25} {SS_cond:>10.4f} {df_cond:>5} {MS_cond:>10.4f} {F2_obs:>10.4f} {F2_crit:>10.4f}")
print(f"{'Субъекты (пациенты)':<25} {SS_subj:>10.4f} {df_subj:>5} {'—':>10} {'—':>10} {'—':>10}")
print(f"{'Ошибка':<25} {SS_error:>10.4f} {df_error:>5} {MS_error:>10.4f} {'—':>10} {'—':>10}")
print(f"{'Итого':<25} {SS_total:>10.4f} {df_total:>5} {'—':>10} {'—':>10} {'—':>10}")

print(f"\np-value = {p2_value:.2e}")
if p2_value < 0.05:
    print("Вывод: H₀ ОТВЕРГАЕТСЯ — время лечения значимо влияет на АД (p < 0,05).")
else:
    print("Вывод: H₀ не отвергается — значимого влияния не обнаружено.")
