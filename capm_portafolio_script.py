import yfinance as yf
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

# 1) Definición de activos y descarga de datos (2 años)
symbols     = ['VTI','SPHD',                           #2 ETF'S
               'AAPL','JNJ','KO','PG','MSFT','PEP',    #6 ACCIONES
               'GC=F','KC=F']                          #2 FUTUROS
today       = datetime.date.today()
start_date  = today - datetime.timedelta(days=365)
prices      = yf.download(symbols, start=start_date, end=today)['Close']

# 2) Retornos diarios y anualizados
returns     = prices.pct_change().dropna()
mu_daily    = returns.mean()
cov_daily   = returns.cov()
mu_annual   = mu_daily * 250
cov_annual  = cov_daily * 250

# 3) Objetivo de retorno: 25% anual
target_return = 0.25

# 4) Optimización
n = len(symbols)
initial_w = np.ones(n) / n
bounds    = [(0.01, 1) for _ in symbols]
constraints = [
    {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
    {'type': 'eq', 'fun': lambda w: np.dot(w, mu_annual) - target_return}
]

def portfolio_variance(w):
    return np.dot(w, np.dot(cov_annual, w))

res = minimize(portfolio_variance,
               x0=initial_w,
               bounds=bounds,
               constraints=constraints)

w_target     = res.x
ret_target   = np.dot(w_target, mu_annual)
vol_target   = np.sqrt(np.dot(w_target, np.dot(cov_annual, w_target)))
sharpe_target = ret_target / vol_target

# 5) Serie diaria y VaR Monte Carlo
ret_pf_daily = returns.dot(w_target)
simulations  = np.random.default_rng(42)     .multivariate_normal(mu_daily, cov_daily, size=10000)     .dot(w_target)
var95_mc      = np.percentile(simulations, 5)

# 6) Histograma de retornos diarios
plt.figure()
plt.hist(ret_pf_daily, bins=40, density=True, alpha=0.6, label='Histograma')
pd.Series(ret_pf_daily).plot(kind='kde', label='KDE')
plt.axvline(ret_pf_daily.mean(), color='black', linestyle='--', label='Media diaria')
plt.axvline(-var95_mc, color='red', linestyle='-', label='VaR 95% MC')
plt.title('Distribución de Retornos Diarios (Portafolio 25%)')
plt.xlabel('Retorno diario')
plt.ylabel('Densidad')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7) Simulación GBM
n_sims = 100
n_days = 125
μ = ret_pf_daily.mean()
σ = ret_pf_daily.std()
drift = μ - 0.5 * σ**2
Z = np.random.default_rng(42).standard_normal(size=(n_days, n_sims))
steps = np.exp(drift + σ * Z)
paths = steps.cumprod(axis=0)

valor_inicial = prices.iloc[-1] @ w_target
paths_real = valor_inicial * paths

plt.figure(figsize=(8,4))
for i in range(n_sims):
    plt.plot(paths_real[:, i], lw=1, alpha=0.5)
plt.title('Simulaciones GBM del Portafolio (Valor Real, 6 meses)')
plt.xlabel('Día')
plt.ylabel('Valor del Portafolio (USD)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 8) Resultados clave
print("=== Portafolio Objetivo 25% ===")
print(f"Rentabilidad anual esperada: {ret_target:.2%}")
print(f"Riesgo anual (σ):             {vol_target:.2%}")
print(f"Sharpe ratio:                 {sharpe_target:.2f}\n")
print("Pesos óptimos (>=1%):")
for sym, w in zip(symbols, w_target):
    print(f"  {sym}: {w:.3f}")

# 9) VaR Paramétrico
mean_daily = ret_pf_daily.mean()
std_daily = ret_pf_daily.std()

var95_d = norm.ppf(0.05, mean_daily, std_daily)
var99_d = norm.ppf(0.01, mean_daily, std_daily)

mean_week = mean_daily * 5
std_week = std_daily * np.sqrt(5)
var95_w = norm.ppf(0.05, mean_week, std_week)
var99_w = norm.ppf(0.01, mean_week, std_week)

mean_month = mean_daily * 21
std_month = std_daily * np.sqrt(21)
var95_m = norm.ppf(0.05, mean_month, std_month)
var99_m = norm.ppf(0.01, mean_month, std_month)

print("\n=== VaR Paramétrico ===")
print(f"Diario  VaR 95%: {var95_d:.4%}, VaR 99%: {var99_d:.4%}")
print(f"Semanal VaR 95%: {var95_w:.4%}, VaR 99%: {var99_w:.4%}")
print(f"Mensual VaR 95%: {var95_m:.4%}, VaR 99%: {var99_m:.4%}")

# 10) CAPM - Beta
benchmark = '^GSPC'
benchmark_prices = yf.download(benchmark, start=start_date, end=today)['Close']
benchmark_returns = benchmark_prices.pct_change().dropna()

aligned_returns = pd.concat([ret_pf_daily, benchmark_returns], axis=1).dropna()
aligned_returns.columns = ['portfolio', 'benchmark']
cov_matrix = np.cov(aligned_returns['portfolio'], aligned_returns['benchmark'])
beta = cov_matrix[0, 1] / cov_matrix[1, 1]

print(f"\n=== CAPM ===")
print(f"Beta del portafolio respecto al S&P 500: {beta:.4f}")
