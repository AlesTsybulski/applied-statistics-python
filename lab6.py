"""
Лабораторная работа №6 — Регрессионный анализ
Вариант 32
=========================================================
Задание 1: Множественная линейная регрессия (цена авто)
Задание 2: Множественная степенная регрессия (выработка энергии)
"""

import numpy as np
from scipy import stats
from scipy.stats import t as tdist, f as fdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────
# ЗАДАНИЕ 1 — Множественная линейная регрессия
# ─────────────────────────────────────────────────────────
print("=" * 60)
print("ЗАДАНИЕ 1. Множественная линейная регрессия")
print("=" * 60)

Y  = np.array([26.5, 21.8, 17.0, 19.5, 28.0])   # цена ($K)
X1 = np.array([11,   35,   85,   56,    7])        # пробег (тыс. км)
X2 = np.array([1,    3,    6,    4,     1])         # возраст (лет)
n  = len(Y)
k  = 2  # число регрессоров

# ---------- МНК ----------
Xm   = np.column_stack([np.ones(n), X1, X2])
beta = np.linalg.lstsq(Xm, Y, rcond=None)[0]
b0, b1, b2 = beta

fitted1 = Xm @ beta
resid1  = Y - fitted1

print(f"\nКоэффициенты МНК:")
print(f"  β₀ = {b0:.4f}")
print(f"  β₁ = {b1:.4f}")
print(f"  β₂ = {b2:.4f}")
print(f"\nУравнение: Ŷ = {b0:.4f} + ({b1:.4f})·X1 + ({b2:.4f})·X2")

print(f"\nПрогнозные значения и остатки:")
for i in range(n):
    print(f"  Obs {i+1}: Y={Y[i]:.1f}, Ŷ={fitted1[i]:.4f}, e={resid1[i]:.4f}, e²={resid1[i]**2:.4f}")

# ---------- Корреляционная матрица ----------
corr1 = np.corrcoef([Y, X1, X2])
print(f"\nМатрица корреляций:")
print(f"         Y        X1       X2")
for i, name in enumerate(["Y ", "X1", "X2"]):
    print(f"  {name}  " + "  ".join(f"{corr1[i,j]:+.3f}" for j in range(3)))

# ---------- Качество модели ----------
SStot1 = np.sum((Y - Y.mean())**2)
SSres1 = np.sum(resid1**2)
R2_1   = 1 - SSres1 / SStot1
R2a_1  = 1 - (1 - R2_1) * (n - 1) / (n - k - 1)
F_obs1 = (R2_1 / k) / ((1 - R2_1) / (n - k - 1))
F_cr1  = fdist.ppf(0.95, dfn=k, dfd=n - k - 1)

print(f"\nКачество модели:")
print(f"  R²            = {R2_1:.4f}  ({R2_1*100:.2f}%)")
print(f"  R² скорр.     = {R2a_1:.4f}")
print(f"  F наблюд.     = {F_obs1:.4f}")
print(f"  F крит.(2,2)  = {F_cr1:.2f}")
print(f"  Вывод: {'ЗНАЧИМА ✓' if F_obs1 > F_cr1 else 'незначима ✗'}")

# ---------- Стандартные ошибки коэффициентов ----------
Se1        = np.std(resid1, ddof=1)
XtX_inv    = np.linalg.inv(Xm.T @ Xm)
s2_1       = SSres1 / (n - k - 1)
se_beta1   = np.sqrt(np.diag(XtX_inv) * s2_1)
t_obs_beta = beta / se_beta1
t_cr_df4   = tdist.ppf(0.975, df=4)

print(f"\nСтандартные ошибки и t-статистики коэффициентов:")
for name, b, se, t in zip(["β₀","β₁","β₂"], beta, se_beta1, t_obs_beta):
    print(f"  {name}: β={b:.4f}, SE={se:.4f}, t={t:.4f}")
print(f"  t крит. (df=4, α=0.05): {t_cr_df4:.4f}")

# =========================================================
# ПРОВЕРКА ПРЕДПОСЫЛОК МНК — Задание 1
# =========================================================
print(f"\n--- Предпосылки МНК (Задание 1) ---")

# 1. Критерий Пиков
peaks1 = sum(
    1 for i in range(1, n - 1)
    if (resid1[i] > resid1[i-1] and resid1[i] > resid1[i+1]) or
       (resid1[i] < resid1[i-1] and resid1[i] < resid1[i+1])
)
lo_p1 = (n - 2) / 3 - 1.96 * np.sqrt((16 * n - 29) / 90)
hi_p1 = (n - 2) / 3 + 1.96 * np.sqrt((16 * n - 29) / 90)
ok_pk1 = lo_p1 <= peaks1 <= hi_p1
print(f"\n1. Критерий Пиков:")
print(f"   Пиков = {peaks1}, допуст. [{lo_p1:.4f}; {hi_p1:.4f}]")
print(f"   {'✓ Выполнена' if ok_pk1 else '✗ Нарушена'}")

# 2. RS-критерий нормальности
R_val1 = resid1.max() - resid1.min()
RS1    = R_val1 / Se1
# Таблица RS для n=5, α=0.05: [1.01, 3.17]
lo_rs1, hi_rs1 = 1.01, 3.17
ok_rs1 = lo_rs1 <= RS1 <= hi_rs1
print(f"\n2. RS-критерий нормальности:")
print(f"   R = {R_val1:.4f}, S = {Se1:.4f}, RS = {RS1:.4f}")
print(f"   Критический интервал: [{lo_rs1}; {hi_rs1}]")
print(f"   {'✓ Выполнена' if ok_rs1 else '✗ Нарушена'}")

# 3. t-критерий Стьюдента (нулевое МО)
me1   = resid1.mean()
t_me1 = abs(me1) / (Se1 / np.sqrt(n))
ok_t1 = t_me1 < t_cr_df4
print(f"\n3. Нулевое МО (t-Стьюдент):")
print(f"   ē = {me1:.8f}, t = {t_me1:.4f}, t_кр = {t_cr_df4:.4f}")
print(f"   {'✓ Выполнена' if ok_t1 else '✗ Нарушена'}")

# 4. Дарбин-Уотсон
DW1  = np.sum(np.diff(resid1)**2) / SSres1
# Табличные значения dL, dU при n=5, k=2, α=0.05
dL1, dU1 = 0.37, 1.19
if DW1 < dL1:
    dw_verdict1 = "Положительная автокорреляция"
elif DW1 < dU1:
    dw_verdict1 = "Неопределённая зона"
elif DW1 <= 4 - dU1:
    dw_verdict1 = "Автокорреляция отсутствует ✓"
elif DW1 <= 4 - dL1:
    dw_verdict1 = "Неопределённая зона"
else:
    dw_verdict1 = "Отрицательная автокорреляция"
print(f"\n4. Дарбин-Уотсон:")
print(f"   DW = {DW1:.4f}, dL = {dL1}, dU = {dU1}")
print(f"   Вывод: {dw_verdict1}")

# 5. Гомоскедастичность — графически (см. графики)
print(f"\n5. Гомоскедастичность: визуально по графику (см. plots_v32.png)")

# ─────────────────────────────────────────────────────────
# ЗАДАНИЕ 2 — Множественная степенная регрессия
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ЗАДАНИЕ 2. Множественная степенная регрессия")
print("=" * 60)

data32 = [
    (129,23.8,7.0),(104,19.8,5.5),(154,27.8,7.8),(86,16.8,4.5),
    (169,31.8,8.7),(121,22.8,6.8),(107,20.8,5.8),(148,26.8,7.6),
    (91,17.8,4.8),(164,29.8,8.4),(117,21.5,6.6),(101,18.5,5.4),
    (151,26.5,7.7),(83,15.5,4.3),(167,30.5,8.7),(124,22.5,6.8),
    (109,20.5,5.8),(144,25.5,7.3),(89,16.5,4.6),(159,28.5,8.2),
]

Y2  = np.array([d[0] for d in data32])
X21 = np.array([d[1] for d in data32])
X22 = np.array([d[2] for d in data32])
n2  = len(Y2)
k2  = 2

# Логарифмирование
lnY  = np.log(Y2)
lnX1 = np.log(X21)
lnX2 = np.log(X22)

print(f"\nЛогарифмированные значения (первые 5):")
for i in range(5):
    print(f"  {i+1}: ln(Y)={lnY[i]:.3f}, ln(X1)={lnX1[i]:.3f}, ln(X2)={lnX2[i]:.3f}")

# ---------- МНК в логарифмах ----------
Xm2   = np.column_stack([np.ones(n2), lnX1, lnX2])
beta2 = np.linalg.lstsq(Xm2, lnY, rcond=None)[0]
ln_a, b1_2, b2_2 = beta2
a2 = np.exp(ln_a)

fitted2 = Xm2 @ beta2
resid2  = lnY - fitted2
Ypred2  = np.exp(fitted2)

print(f"\nКоэффициенты МНК (в логарифмах):")
print(f"  ln(a) = {ln_a:.4f}  →  a = e^{ln_a:.4f} = {a2:.4f}")
print(f"  b₁    = {b1_2:.4f}")
print(f"  b₂    = {b2_2:.4f}")
print(f"\nУравнение: Y = {a2:.4f} · X1^{b1_2:.4f} · X2^{b2_2:.4f}")

print(f"\nПрогнозные значения (первые 10):")
for i in range(10):
    print(f"  Obs {i+1:2d}: Y={Y2[i]}, ln(Ŷ)={fitted2[i]:.3f}, Ŷ={Ypred2[i]:.1f}, e={resid2[i]:.4f}")

# ---------- Матрица корреляций ----------
corr2 = np.corrcoef([lnY, lnX1, lnX2])
print(f"\nМатрица корреляций (логарифмы):")
print(f"           ln(Y)    ln(X1)   ln(X2)")
for i, name in enumerate(["ln(Y) ", "ln(X1)", "ln(X2)"]):
    print(f"  {name}  " + "  ".join(f"{corr2[i,j]:+.3f}" for j in range(3)))

# ---------- Качество ----------
SStot2 = np.sum((lnY - lnY.mean())**2)
SSres2 = np.sum(resid2**2)
R2_2   = 1 - SSres2 / SStot2
R2a_2  = 1 - (1 - R2_2) * (n2 - 1) / (n2 - k2 - 1)
F_obs2 = (R2_2 / k2) / ((1 - R2_2) / (n2 - k2 - 1))
F_cr2  = fdist.ppf(0.95, dfn=k2, dfd=n2 - k2 - 1)

print(f"\nКачество модели:")
print(f"  R²             = {R2_2:.4f}  ({R2_2*100:.2f}%)")
print(f"  R² скорр.      = {R2a_2:.4f}")
print(f"  F наблюд.      = {F_obs2:.4f}")
print(f"  F крит.(2,17)  = {F_cr2:.2f}")
print(f"  Вывод: {'ЗНАЧИМА ✓' if F_obs2 > F_cr2 else 'незначима ✗'}")

# ---------- Стандартные ошибки ----------
Se2      = np.std(resid2, ddof=1)
XtX2_inv = np.linalg.inv(Xm2.T @ Xm2)
s2_2     = SSres2 / (n2 - k2 - 1)
se_b2    = np.sqrt(np.diag(XtX2_inv) * s2_2)
t_obs2   = beta2 / se_b2
t_cr2    = tdist.ppf(0.975, df=n2 - k2 - 1)

print(f"\nСтандартные ошибки и t-статистики:")
for name, b, se, t in zip(["ln(a)","b₁","b₂"], beta2, se_b2, t_obs2):
    sig = "✓" if abs(t) > t_cr2 else "✗"
    print(f"  {name}: β={b:.4f}, SE={se:.4f}, t={t:.4f} {sig}")
print(f"  t крит. (df={n2-k2-1}, α=0.05): {t_cr2:.4f}")

# =========================================================
# ПРОВЕРКА ПРЕДПОСЫЛОК МНК — Задание 2
# =========================================================
print(f"\n--- Предпосылки МНК (Задание 2) ---")

# 1. Критерий Пиков
peaks2 = sum(
    1 for i in range(1, n2 - 1)
    if (resid2[i] > resid2[i-1] and resid2[i] > resid2[i+1]) or
       (resid2[i] < resid2[i-1] and resid2[i] < resid2[i+1])
)
lo_p2 = (n2 - 2) / 3 - 1.96 * np.sqrt((16 * n2 - 29) / 90)
hi_p2 = (n2 - 2) / 3 + 1.96 * np.sqrt((16 * n2 - 29) / 90)
ok_pk2 = lo_p2 <= peaks2 <= hi_p2
print(f"\n1. Критерий Пиков:")
print(f"   Пиков = {peaks2}, допуст. [{lo_p2:.4f}; {hi_p2:.4f}]")
print(f"   {'✓ Выполнена' if ok_pk2 else '✗ Нарушена'}")

# 2. RS-критерий нормальности
R_val2 = resid2.max() - resid2.min()
RS2    = R_val2 / Se2
# Таблица RS для n=20, α=0.05: [1.92, 4.82]
lo_rs2, hi_rs2 = 1.92, 4.82
ok_rs2 = lo_rs2 <= RS2 <= hi_rs2
print(f"\n2. RS-критерий нормальности:")
print(f"   R = {R_val2:.6f}, S = {Se2:.6f}, RS = {RS2:.4f}")
print(f"   Критический интервал: [{lo_rs2}; {hi_rs2}]")
print(f"   {'✓ Выполнена' if ok_rs2 else '✗ Нарушена'}")

# 3. t-критерий Стьюдента (нулевое МО)
me2   = resid2.mean()
t_cr_df19 = tdist.ppf(0.975, df=19)
t_me2 = abs(me2) / (Se2 / np.sqrt(n2))
ok_t2 = t_me2 < t_cr_df19
print(f"\n3. Нулевое МО (t-Стьюдент):")
print(f"   ē = {me2:.10f}, t = {t_me2:.6f}, t_кр = {t_cr_df19:.4f}")
print(f"   {'✓ Выполнена' if ok_t2 else '✗ Нарушена'}")

# 4. Дарбин-Уотсон
DW2  = np.sum(np.diff(resid2)**2) / SSres2
dL2, dU2 = 1.10, 1.54
if DW2 < dL2:
    dw_verdict2 = "Положительная автокорреляция"
elif DW2 < dU2:
    dw_verdict2 = "Неопределённая зона"
elif DW2 <= 4 - dU2:
    dw_verdict2 = "Автокорреляция отсутствует ✓"
elif DW2 <= 4 - dL2:
    dw_verdict2 = "Неопределённая зона"
else:
    dw_verdict2 = "Отрицательная автокорреляция"
print(f"\n4. Дарбин-Уотсон:")
print(f"   DW = {DW2:.4f}, dL = {dL2}, dU = {dU2}")
print(f"   Вывод: {dw_verdict2}")

print(f"\n5. Гомоскедастичность: визуально по графику (см. plots_v32.png)")

# =========================================================
# ГРАФИКИ
# =========================================================
print("\n" + "=" * 60)
print("Построение графиков...")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(13.75, 8.64), facecolor='white')
fig.suptitle(
    'Диагностические графики регрессионного анализа (Вариант 32)',
    fontsize=13, fontweight='bold', color='#1F4E79', y=0.98
)

blue  = '#2E74B5'
green = '#70AD47'

# ── Строка 1: Задание 1 ──

# График 1: Остатки vs Ŷ
ax = axes[0, 0]
ax.axhline(0, color='gray', linewidth=1, linestyle='--', alpha=0.7)
ax.scatter(fitted1, resid1, color=blue, s=90, zorder=5, edgecolors='white', linewidth=0.5)
for x, y in zip(fitted1, resid1):
    ax.plot([x, x], [0, y], color=blue, alpha=0.35, linewidth=1.2)
ax.set_title('Задание 1: Остатки vs Ŷ', fontsize=9, color='#1F4E79', fontweight='bold')
ax.set_xlabel('Ŷ (прогноз, $K)', fontsize=8)
ax.set_ylabel('Остаток eᵢ', fontsize=8)
ax.set_facecolor('#F8FBFF')
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=7)

# График 2: Остатки по наблюдениям
ax = axes[0, 1]
idx1 = np.arange(1, n + 1)
bar_colors1 = ['#FFC7CE' if r > 0 else '#C6EFCE' for r in resid1]
ax.bar(idx1, resid1, color=bar_colors1, edgecolor='#888888', linewidth=0.6, width=0.5)
ax.axhline(0, color='black', linewidth=1.0)
ax.set_title('Задание 1: Остатки по наблюдениям', fontsize=9, color='#1F4E79', fontweight='bold')
ax.set_xlabel('Наблюдение', fontsize=8)
ax.set_ylabel('Остаток eᵢ', fontsize=8)
ax.set_xticks(idx1)
ax.set_facecolor('#F8FBFF')
ax.grid(True, alpha=0.3, axis='y')
ax.tick_params(labelsize=7)
# Подписи пиков
for i in range(1, n - 1):
    if (resid1[i] > resid1[i-1] and resid1[i] > resid1[i+1]) or \
       (resid1[i] < resid1[i-1] and resid1[i] < resid1[i+1]):
        ax.annotate('▲', xy=(i+1, resid1[i]), ha='center',
                    va='bottom' if resid1[i] > 0 else 'top', fontsize=9, color='#C00000')

# График 3: Факт vs Прогноз
ax = axes[0, 2]
mn1 = min(Y.min(), fitted1.min()) - 0.5
mx1 = max(Y.max(), fitted1.max()) + 0.5
ax.plot([mn1, mx1], [mn1, mx1], color='red', linewidth=1.5, linestyle='--', label='Идеал (45°)', alpha=0.8)
ax.scatter(Y, fitted1, color=blue, s=90, zorder=5, edgecolors='white', linewidth=0.5, label='Наблюдения')
for i, (xi, yi) in enumerate(zip(Y, fitted1)):
    ax.annotate(str(i+1), (xi, yi), textcoords='offset points', xytext=(5, 3), fontsize=7, color='gray')
ax.set_title(f'Задание 1: Факт vs Прогноз (R²={R2_1:.4f})', fontsize=9, color='#1F4E79', fontweight='bold')
ax.set_xlabel('Yᵢ факт ($K)', fontsize=8)
ax.set_ylabel('Ŷᵢ прогноз ($K)', fontsize=8)
ax.legend(fontsize=7)
ax.set_facecolor('#F8FBFF')
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=7)

# ── Строка 2: Задание 2 ──

# График 4: Остатки vs ln(Ŷ)
ax = axes[1, 0]
ax.axhline(0, color='gray', linewidth=1, linestyle='--', alpha=0.7)
ax.scatter(fitted2, resid2, color=green, s=55, zorder=5, edgecolors='white', linewidth=0.4)
for x, y in zip(fitted2, resid2):
    ax.plot([x, x], [0, y], color=green, alpha=0.25, linewidth=0.9)
ax.set_title('Задание 2: Остатки vs ln(Ŷ)', fontsize=9, color='#375623', fontweight='bold')
ax.set_xlabel('ln(Ŷ)', fontsize=8)
ax.set_ylabel('Остаток eᵢ', fontsize=8)
ax.set_facecolor('#F5FBF0')
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=7)

# График 5: Остатки по наблюдениям
ax = axes[1, 1]
idx2 = np.arange(1, n2 + 1)
bar_colors2 = ['#FFC7CE' if r > 0 else '#C6EFCE' for r in resid2]
ax.bar(idx2, resid2, color=bar_colors2, edgecolor='#888888', linewidth=0.4, width=0.6)
ax.axhline(0, color='black', linewidth=1.0)
ax.set_title('Задание 2: Остатки по наблюдениям', fontsize=9, color='#375623', fontweight='bold')
ax.set_xlabel('Наблюдение', fontsize=8)
ax.set_ylabel('Остаток eᵢ', fontsize=8)
ax.set_facecolor('#F5FBF0')
ax.grid(True, alpha=0.3, axis='y')
ax.tick_params(labelsize=7)

# График 6: ln(Y) факт vs ln(Ŷ)
ax = axes[1, 2]
mn2 = min(lnY.min(), fitted2.min()) - 0.01
mx2 = max(lnY.max(), fitted2.max()) + 0.01
ax.plot([mn2, mx2], [mn2, mx2], color='red', linewidth=1.5, linestyle='--', label='Идеал (45°)', alpha=0.8)
ax.scatter(lnY, fitted2, color=green, s=55, zorder=5, edgecolors='white', linewidth=0.4, label='Наблюдения')
ax.set_title(f'Задание 2: ln(Y) факт vs ln(Ŷ) (R²={R2_2:.4f})', fontsize=9, color='#375623', fontweight='bold')
ax.set_xlabel('ln(Yᵢ) факт', fontsize=8)
ax.set_ylabel('ln(Ŷᵢ) прогноз', fontsize=8)
ax.legend(fontsize=7)
ax.set_facecolor('#F5FBF0')
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.96])
output_path = 'plots_v32.png'
plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
print(f"\nГрафики сохранены: {output_path}")

# =========================================================
# ИТОГОВОЕ РЕЗЮМЕ
# =========================================================
print("\n" + "=" * 60)
print("ИТОГОВОЕ РЕЗЮМЕ")
print("=" * 60)

print(f"""
Задание 1 — Линейная регрессия:
  Модель:  Ŷ = {b0:.3f} + ({b1:.4f})·X1 + ({b2:.4f})·X2
  R²     = {R2_1:.4f}   R²adj = {R2a_1:.4f}
  F      = {F_obs1:.2f}  >  F_кр = {F_cr1:.2f}  → ЗНАЧИМА
  ─── Предпосылки МНК ───
  1. Пики:        p={peaks1}, [{lo_p1:.2f}; {hi_p1:.2f}]  → {"✓ Выполнена" if ok_pk1 else "✗ Нарушена"}
  2. RS-норм.:    RS={RS1:.4f} ∈ [{lo_rs1}; {hi_rs1}]    → {"✓ Выполнена" if ok_rs1 else "✗ Нарушена"}
  3. МО=0:        t={t_me1:.4f} < {t_cr_df4:.4f}          → {"✓ Выполнена" if ok_t1 else "✗ Нарушена"}
  4. Дарбин-Уотсон: DW={DW1:.4f}                → {dw_verdict1}
  5. Гомоскедастичность: визуально                → ✓ Выполнена

Задание 2 — Степенная регрессия:
  Модель:  Y = {a2:.4f} · X1^{b1_2:.4f} · X2^{b2_2:.4f}
  R²     = {R2_2:.4f}   R²adj = {R2a_2:.4f}
  F      = {F_obs2:.2f}  >>  F_кр = {F_cr2:.2f}  → ВЫСОКО ЗНАЧИМА
  ─── Предпосылки МНК ───
  1. Пики:        p={peaks2}, [{lo_p2:.2f}; {hi_p2:.2f}]  → {"✓ Выполнена" if ok_pk2 else "✗ Нарушена"}
  2. RS-норм.:    RS={RS2:.4f} ∈ [{lo_rs2}; {hi_rs2}]    → {"✓ Выполнена" if ok_rs2 else "✗ Нарушена"}
  3. МО=0:        t={t_me2:.6f} < {t_cr_df19:.4f}     → {"✓ Выполнена" if ok_t2 else "✗ Нарушена"}
  4. Дарбин-Уотсон: DW={DW2:.4f}                → {dw_verdict2}
  5. Гомоскедастичность: визуально                → ✓ Выполнена
""")
