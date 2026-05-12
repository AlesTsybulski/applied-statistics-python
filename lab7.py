import numpy as np
from scipy import stats

# ── ДАННЫЕ (Вариант 15) ──────────────────────────────────────────
Q1 = [4, 5, 3, 4, 5, 3, 4, 5, 4, 5]
Q2 = [2, 1, 2, 1, 2, 1, 2, 1, 2, 2]
Q3 = [5, 5, 3, 4, 5, 4, 4, 5, 4, 5]
Q4 = [2, 1, 2, 1, 2, 1, 2, 2, 1, 2]
Q5 = [4, 5, 3, 4, 5, 4, 4, 5, 4, 5]

data = np.array([Q1, Q2, Q3, Q4, Q5], dtype=float).T  # форма: (10, 5)
n, p = data.shape  # n=10 наблюдений, p=5 переменных
cols = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

print("=" * 60)
print("ЛАБОРАТОРНАЯ РАБОТА №7 — ВАРИАНТ 32")
print("=" * 60)

# ════════════════════════════════════════════════════════════════
# 1. СРЕДНИЕ И ДИСПЕРСИИ
# ════════════════════════════════════════════════════════════════
print("\n── 1. СРЕДНИЕ И ДИСПЕРСИИ ──")
for i, q in enumerate(cols):
    mean = np.mean(data[:, i])
    var  = np.var(data[:, i], ddof=1)
    print(f"  {q}: среднее = {mean:.4f},  дисперсия = {var:.4f}")

# ════════════════════════════════════════════════════════════════
# 2. МАТРИЦА КОРРЕЛЯЦИЙ
# ════════════════════════════════════════════════════════════════
R = np.corrcoef(data.T)

print("\n── 2. МАТРИЦА КОРРЕЛЯЦИЙ R ──")
print("       " + "  ".join(f"{c:>7}" for c in cols))
for i in range(p):
    row = "  ".join(f"{R[i, j]:7.3f}" for j in range(p))
    print(f"  {cols[i]}  {row}")

# ════════════════════════════════════════════════════════════════
# 3. ТЕСТ БАРТЛЕТТА
# ════════════════════════════════════════════════════════════════
det_R  = np.linalg.det(R)
ln_det = np.log(det_R)
coeff  = n - 1 - (2 * p + 5) / 6
chi2   = -coeff * ln_det
df     = p * (p - 1) // 2
p_val  = 1 - stats.chi2.cdf(chi2, df)

print("\n── 3. ТЕСТ БАРТЛЕТТА ──")
print(f"  |R|          = {det_R:.6f}")
print(f"  ln|R|        = {ln_det:.4f}")
print(f"  коэффициент  = {coeff}")
print(f"  χ²           = {chi2:.4f}")
print(f"  df           = {df}")
print(f"  p-value      = {p_val:.4f}")
print(f"  Вывод: {'H₀ отвергается ✓' if p_val < 0.05 else 'H₀ не отвергается ✗'}")

# ════════════════════════════════════════════════════════════════
# 4. МЕРА KMO
# ════════════════════════════════════════════════════════════════
R_inv = np.linalg.inv(R)

# Частные корреляции
partial = np.zeros((p, p))
for i in range(p):
    for j in range(p):
        if i != j:
            partial[i, j] = -R_inv[i, j] / np.sqrt(R_inv[i, i] * R_inv[j, j])

sum_r2 = sum(R[i, j] ** 2 for i in range(p) for j in range(p) if i != j)
sum_a2 = sum(partial[i, j] ** 2 for i in range(p) for j in range(p) if i != j)
KMO = sum_r2 / (sum_r2 + sum_a2)

print("\n── 4. МЕРА KMO ──")
print(f"  Σ r²ᵢⱼ (i≠j) = {sum_r2:.4f}")
print(f"  Σ a²ᵢⱼ (i≠j) = {sum_a2:.4f}")
print(f"  KMO           = {KMO:.4f}")
for i in range(p):
    sr2 = sum(R[i, j] ** 2 for j in range(p) if j != i)
    sa2 = sum(partial[i, j] ** 2 for j in range(p) if j != i)
    print(f"  KMO({cols[i]})      = {sr2 / (sr2 + sa2):.3f}")
print(f"  Вывод: {'допустимо ✓' if KMO > 0.5 else 'недопустимо ✗'} (KMO > 0.5)")

# ════════════════════════════════════════════════════════════════
# 5. СОБСТВЕННЫЕ ЗНАЧЕНИЯ
# ════════════════════════════════════════════════════════════════
eigenvalues, eigenvectors = np.linalg.eigh(R)

# eigh возвращает по возрастанию — разворачиваем
idx = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("\n── 5. СОБСТВЕННЫЕ ЗНАЧЕНИЯ ──")
print(f"  {'№':<4} {'λ':>8}  {'%дисп':>8}  {'накопл.':>9}  {'отбор':>8}")
cumvar = 0
m = 0  # число факторов с λ > 1
for i, lam in enumerate(eigenvalues):
    pct    = lam / p * 100
    cumvar += pct
    keep   = "> 1 ✓" if lam > 1 else "< 1 ✗"
    if lam > 1:
        m += 1
    print(f"  {i+1:<4} {lam:>8.4f}  {pct:>7.2f}%  {cumvar:>8.2f}%  {keep:>8}")
print(f"  Отобрано факторов: {m}")

# ════════════════════════════════════════════════════════════════
# 6. НЕПОВЁРНУТЫЕ ФАКТОРНЫЕ НАГРУЗКИ
# ════════════════════════════════════════════════════════════════
L_unrot = eigenvectors[:, :m] * np.sqrt(eigenvalues[:m])

print("\n── 6. НЕПОВЁРНУТЫЕ НАГРУЗКИ (L = E · √Λ) ──")
header = "  " + "  ".join(f"{'Ф'+str(k+1):>9}" for k in range(m))
print(header)
for i in range(p):
    row = "  ".join(f"{L_unrot[i, k]:9.3f}" for k in range(m))
    print(f"  {cols[i]}  {row}")

# ════════════════════════════════════════════════════════════════
# 7. ВРАЩЕНИЕ VARIMAX
# ════════════════════════════════════════════════════════════════
def varimax(Phi, gamma=1.0, max_iter=1000, tol=1e-8):
    """Аналитическое вращение Varimax."""
    p2, k = Phi.shape
    R2 = np.eye(k)
    d  = 0
    for _ in range(max_iter):
        d_old  = d
        Lambda = Phi @ R2
        u = Lambda ** 3 - (gamma / p2) * Lambda @ np.diag(np.sum(Lambda ** 2, axis=0))
        U, S, Vt = np.linalg.svd(Phi.T @ u)
        R2 = U @ Vt
        d  = np.sum(S)
        if d_old != 0 and abs(d / d_old - 1) < tol:
            break
    return Phi @ R2

L_rot = varimax(L_unrot)

print("\n── 7. НАГРУЗКИ ПОСЛЕ ВРАЩЕНИЯ VARIMAX ──")
print("  " + "  ".join(f"{'Ф'+str(k+1):>9}" for k in range(m)) + "    h² (коммун-ть)")
for i in range(p):
    row  = "  ".join(f"{L_rot[i, k]:9.3f}" for k in range(m))
    comm = sum(L_rot[i, k] ** 2 for k in range(m))
    flag = "  ← входит" if any(abs(L_rot[i, k]) > 0.5 for k in range(m)) else ""
    print(f"  {cols[i]}  {row}    {comm:.3f}{flag}")

print("\n  Интерпретация факторов (нагрузка > 0.5):")
for k in range(m):
    members = [cols[i] for i in range(p) if abs(L_rot[i, k]) > 0.5]
    print(f"  Фактор {k+1}: {', '.join(members)}")

# ════════════════════════════════════════════════════════════════
# 8. АЛЬФА КРОНБАХА
# ════════════════════════════════════════════════════════════════
def cronbach_alpha(items):
    """
    items: массив (n_наблюд × k_вопросов)
    Возвращает α Кронбаха.
    """
    k         = items.shape[1]
    item_vars = np.var(items, ddof=1, axis=0)
    total_var = np.var(items.sum(axis=1), ddof=1)
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)

# Фактор 1: Q1, Q3, Q5  (индексы 0, 2, 4)
alpha1 = cronbach_alpha(data[:, [0, 2, 4]])
# Фактор 2: Q2, Q4      (индексы 1, 3)
alpha2 = cronbach_alpha(data[:, [1, 3]])

print("\n── 8. АЛЬФА КРОНБАХА ──")
print(f"  Фактор 1 (Q1, Q3, Q5): α = {alpha1:.4f}  {'отлично ✓' if alpha1 >= 0.9 else 'хорошо ✓' if alpha1 >= 0.8 else 'приемлемо ✓' if alpha1 >= 0.7 else 'ниже нормы ✗'}")
print(f"  Фактор 2 (Q2, Q4):     α = {alpha2:.4f}  {'отлично ✓' if alpha2 >= 0.9 else 'хорошо ✓' if alpha2 >= 0.8 else 'приемлемо ✓' if alpha2 >= 0.7 else 'ниже нормы ✗'}")

# ════════════════════════════════════════════════════════════════
# 9. ИТОГ
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ИТОГОВАЯ СВОДКА")
print("=" * 60)
checks = [
    ("Тест Бартлетта",        f"χ²={chi2:.2f}, p={p_val:.4f}",  p_val < 0.05),
    ("Мера KMO",              f"KMO={KMO:.3f}",                  KMO > 0.5),
    ("Объяснённая дисперсия", f"{cumvar:.2f}% ({m} фактора)",    cumvar > 60),
    ("Альфа Кронбаха",        f"α₁={alpha1:.3f}, α₂={alpha2:.3f}", alpha1 >= 0.7 and alpha2 >= 0.7),
]
for name, result, ok in checks:
    print(f"  {'✓' if ok else '✗'}  {name:<28} {result}")
