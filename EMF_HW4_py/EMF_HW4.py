# ==========================================================================
# EMPIRICAL METHODS FOR FINANCE
# Homework 4 Modeling Volatility: GARCH Models
#
# Author : Maxime Borel
# Date: August 2020
# ==========================================================================

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import arch
from arch import arch_model
from scipy import stats
from arch.univariate import GARCH
from arch.univariate import ZeroMean

# Qestion 1
df = pd.read_excel('DATA_HW4.xlsx', skiprows=1, index_col='Code', parse_dates=True)
df.values[:, -1] = df.iloc[:, -1] / (52*100)
returns = df.pct_change().dropna()
returns = np.diff(df.values[:, 0:-1], axis=0) / df.values[0:-1, 0:-1]
rf = df.values[1:, -1].reshape((df.values[1:, -1].shape[0], 1))
returns = np.hstack((returns, rf))

# Question 2
avg_return = returns.mean(axis=0)
avg_return = avg_return.reshape(avg_return.shape[0], 1)
cov_return = np.cov(returns[:, :-1].T)
lambda_i = np.array([[2],
                     [10]])
e = np.ones((2, 1))
alpha_2 = np.zeros(avg_return.shape)
alpha_2[:-1] = (1/lambda_i[0])*np.linalg.inv(cov_return)@(avg_return[:-1]-e*avg_return[-1])
alpha_2[-1] = 1-alpha_2.sum()
asset_names = ['Stock', 'Bonds', 'Rf']
alpha_2 = pd.DataFrame(alpha_2, index=asset_names, columns=['Weights'])
alpha_2.to_excel('alpha_2.xlsx')

alpha_10 = np.zeros(avg_return.shape)
alpha_10[:-1] = (1/lambda_i[1])*np.linalg.inv(cov_return)@(avg_return[:-1]-e*avg_return[-1])
alpha_10[-1] = 1-alpha_10.sum()
alpha_10 = pd.DataFrame(alpha_10, index=asset_names, columns=['Weights'])
alpha_10.to_excel('alpha_10.xlsx')

# Question 3
# 3a
ER = returns[:, :-1] - returns[:, -1].reshape(returns.shape[0], 1)
stock_LF = sm.stats.diagnostic.lilliefors(ER[:, 0], dist='norm')
bonds_LF = sm.stats.diagnostic.lilliefors(ER[:, 1], dist='norm')
result_lf = np.zeros((2, 2))
result_lf[:, 0] = stock_LF
result_lf[:, 1] = bonds_LF
result_lf = pd.DataFrame(result_lf, index=['ks stat', 'pvalue'],
                         columns=['Stock', 'Bonds'])
result_lf.to_excel('result_lf.xlsx')

LJB_stock = sm.stats.acorr_ljungbox(ER[:, 0], lags=4, return_df=True)
LJB_bonds = sm.stats.acorr_ljungbox(ER[:, 1], lags=4, return_df=True)
result_LJB = np.hstack((LJB_stock.values, LJB_bonds.values))
result_LJB = pd.DataFrame(result_LJB, index=['1', '2', '3', '4'],
                          columns=['Stock Tstat', 'Stock pvalue', 'Bonds tstat', 'Bonds pval'])
result_LJB.to_excel('result_LJB.xlsx')

# 3b
X_stock = returns[:-1, 0]
X_stock = X_stock.reshape(X_stock.shape[0], 1)
X_stock = sm.add_constant(X_stock)
Y_stock = returns[1:, 0]
X_bonds = returns[:-1, 1]
X_bonds = X_bonds.reshape(X_bonds.shape[0], 1)
X_bonds = sm.add_constant(X_bonds)
Y_bonds = returns[1:, 1]

model_s = sm.OLS(Y_stock, X_stock).fit()
stock_resid = model_s.resid.reshape((model_s.resid.shape[0], 1))

model_b = sm.OLS(Y_bonds, X_bonds).fit()
bonds_resid = model_b.resid.reshape((model_s.resid.shape[0], 1))

result_ar = np.hstack((model_s.params.reshape((model_s.params.shape[0], 1)),
                      model_s.tvalues.reshape((model_s.tvalues.shape[0], 1)),
                      model_b.params.reshape((model_b.params.shape[0], 1)),
                      model_b.tvalues.reshape((model_b.tvalues.shape[0], 1))))
result_ar = pd.DataFrame(result_ar, index=['Alpha', 'Beta'],
                         columns=['Stock param', 'tstat', 'Bonds param', 'tstat'])
result_ar.to_excel('result_ar.xlsx')

# Q3C
y_slmtest = stock_resid[4:]**2
y_slmtest = y_slmtest.reshape((y_slmtest.shape[0], 1))
x_slmtest = np.hstack((stock_resid[3:-1]**2, stock_resid[2:-2]**2,
                       stock_resid[1:-3]**2, stock_resid[0:-4]**2))
x_slmtest = sm.add_constant(x_slmtest)

model_slm = sm.OLS(y_slmtest, x_slmtest).fit()
rsq_s = model_slm.rsquared
lmtest_s = rsq_s * stock_resid.shape[0]

ljb_archs = sm.stats.acorr_ljungbox(stock_resid**2, lags=4, return_df=True)

y_blmtest = bonds_resid[4:]**2
y_blmtest = y_blmtest.reshape((y_blmtest.shape[0], 1))
x_blmtest = np.hstack((bonds_resid[3:-1]**2, bonds_resid[2:-2]**2,
                       bonds_resid[1:-3]**2, bonds_resid[0:-4]**2))
x_blmtest = sm.add_constant(x_blmtest)

model_blm = sm.OLS(y_blmtest, x_blmtest).fit()
rsq_b = model_blm.rsquared
lmtest_b = rsq_b * bonds_resid.shape[0]

cv_lm = stats.chi2.ppf(0.95, df=x_blmtest.shape[1])
pval_slm = stats.chi2.cdf([lmtest_s, lmtest_b], df=y_blmtest.shape[1])

ljb_archb = sm.stats.acorr_ljungbox(bonds_resid**2, lags=4, return_df=True)

result_lm = pd.DataFrame([[lmtest_s, lmtest_b],
                         [1-pval_slm[0], 1-pval_slm[1]],
                         [cv_lm, cv_lm],
                         [ljb_archs.iloc[-1, 0], ljb_archb.iloc[-1, 0]],
                         [ljb_archs.iloc[-1, 1], ljb_archb.iloc[-1, 1]]],
                         index=['tstat', 'pval', 'cv_lm', 'LJB 4 lags', 'pval'],
                         columns=['Stock', 'Bonds'])
result_lm.to_excel('result_lm.xlsx')

# Q3d estimate parameter of the GARCH process
am_s = arch_model(stock_resid, mean='Zero', vol='GARCH', p=1, q=1, dist='normal',
                  power=2, rescale=False)
garch_s = am_s.fit()
print(garch_s.summary())
garch_s_forecast = garch_s.forecast(horizon=252)

am_b = arch_model(bonds_resid, mean="Zero", vol='GARCH', p=1, q=1, dist='normal',
                  power=2, rescale=False)
garch_b = am_b.fit()
print(garch_b.summary())
garch_b_forecast = garch_b.forecast(horizon=252)

# compute the tstat for alpha + beta
#stock
ab_s = garch_s.params.values[1:].sum()
tstat_pers = (ab_s - 1) / np.sqrt(np.diag(garch_s.param_cov.values[1:, 1:]).sum() +
                                  2*garch_s.param_cov.values[-1, 1])
ab_b = garch_b.params.values[1:].sum()
tstat_perb = (ab_b - 1) / np.sqrt(np.diag(garch_b.param_cov.values[1:, 1:]).sum() +
                                  2*garch_b.param_cov.values[-1, 1])

result_garch = np.array([[garch_s.params.append(pd.Series(ab_s, index=['a+b']))],
                [garch_s.tvalues.append(pd.Series(tstat_pers, index=['a+b']))],
                [garch_b.params.append(pd.Series(ab_b, index=['a+b']))],
                [garch_b.tvalues.append(pd.Series(tstat_perb, index=['a+b']))]])
result_garch = result_garch.reshape((result_garch.shape[0], result_garch.shape[0]))
result_garch = pd.DataFrame(result_garch, index=['omega', 'a', 'b', 'a+b'],
                            columns=['stock para', 'stock tstat', 'bonds para', 'bonds tstat'])
result_garch.to_excel('result_garch.xlsx')

# Q3e make forecast
fig1 = plt.figure()
plt.plot(garch_s.conditional_volatility)
fig1.savefig('fig1.png')
plt.show()

n = 252
o_s = garch_s.params.values[0]
a_s = garch_s.params.values[1]
b_s = garch_s.params.values[2]
o_b = garch_b.params.values[0]
a_b = garch_b.params.values[1]
b_b = garch_b.params.values[2]
forecast_sigma = np.zeros((n, 2))
forecast_sigma[0, 0] = o_s + a_s*(stock_resid[-1]**2) + b_s*(garch_s.conditional_volatility[-1]**2)
forecast_sigma[0, 1] = o_b + a_b*(bonds_resid[-1]**2) + b_b*(garch_b.conditional_volatility[-1]**2)

for i in range(1, n):
    forecast_sigma[i, 0] = o_s + (a_s + b_s)*forecast_sigma[i-1, 0]
    forecast_sigma[i, 1] = o_b + (a_b + b_b)*forecast_sigma[i-1, 1]

uncond_vol_s = np.sqrt(o_s / (1 - a_s + b_s))
uncond_vol_b = np.sqrt(o_b / (1 - a_b + b_b))

fig2 = plt.figure()
plt.subplot(1, 2, 1)
plt.plot(np.sqrt(forecast_sigma[:53, 0]))
plt.title('stock')
plt.subplot(1, 2, 2)
plt.plot(np.sqrt(forecast_sigma[:53, 1]))
plt.title('bonds')
fig2.savefig('fig2.png')
plt.show()

fig3 = plt.figure()
plt.subplot(1, 2, 1)
plt.plot(np.sqrt(forecast_sigma[:, 0]))
plt.title('stock')
plt.subplot(1, 2, 2)
plt.plot(np.sqrt(forecast_sigma[:, 1]))
plt.title('bonds')
fig3.savefig('fig3.png')
plt.show()

fig4 = plt.figure()
plt.subplot(1, 2, 1)
plt.plot(np.sqrt(garch_s_forecast.variance.values[-1, :]))
plt.title('Stock volatility')
plt.xlabel('forecast date')
plt.ylabel('% of volatility')
plt.subplot(1, 2, 2)
plt.plot(np.sqrt(garch_b_forecast.variance.values[-1, :]))
plt.title('Bond volatility')
plt.xlabel('forecast date')
fig4.savefig('fig4.png')
plt.show()

# Q4
# Q4a
# compute the fitted value AR and the cond variance and correlation
mu_s = model_s.fittedvalues.reshape((model_s.fittedvalues.shape[0], 1))
mu_b = model_b.fittedvalues.reshape((model_b.fittedvalues.shape[0], 1))
mu_t = np.hstack((mu_s, mu_b))
sigma_s = garch_s.conditional_volatility**2
sigma_b = garch_b.conditional_volatility**2

rho = np.corrcoef(stock_resid, bonds_resid, rowvar=False)
rho = rho[0, 1]
cov_sb = rho*sigma_s*sigma_b
Sigma = np.zeros((mu_b.shape[0], 2, 2))
Sigma[:, 0, 0] = sigma_s
Sigma[:, 1, 1] = sigma_b
Sigma[:, 1, 0] = cov_sb
Sigma[:, 0, 1] = cov_sb

# Q4b
d_alpha_2 = np.zeros((mu_b.shape[0], 3))
d_alpha_10 = np.zeros((mu_b.shape[0], 3))

for i in range(0, mu_b.shape[0]):
    d_alpha_2[i, :-1] = (1/lambda_i[0])*(mu_t[i, :]-e.T*rf[i, :])@np.linalg.inv(Sigma[i, :, :])
    d_alpha_10[i, :-1] = (1/lambda_i[1])*(mu_t[i, :]-e.T*rf[i, :])@np.linalg.inv(Sigma[i, :, :])

d_alpha_2[:, -1] = 1-d_alpha_2.sum(axis=1)
d_alpha_10[:, -1] = 1-d_alpha_10.sum(axis=1)

# Q4c
fig5 = plt.figure()
plt.plot(d_alpha_2)
plt.plot(np.ones((d_alpha_2.shape[0], 3))*alpha_2.values.T)
plt.legend(['Stock d', 'Bonds d', 'Rf d', 'Stock', 'Bonds', 'Rf'])
plt.title('Asset allocation with GARCH with lambda=2')
plt.xlabel('Time')
plt.ylabel('Weights')
fig5.savefig('fig5.png')
plt.show()

fig6 = plt.figure()
plt.plot(d_alpha_10)
plt.plot(np.ones((d_alpha_2.shape[0], 3))*alpha_10.values.T)
plt.legend(['Stock d', 'Bonds d', 'Rf d', 'Stock', 'Bonds', 'Rf'])
plt.title('Asset allocation with GARCH with lambda=10')
plt.xlabel('Time')
plt.ylabel('Weights')
fig6.savefig('fig6.png')
plt.show()

# Q4d
port_ret_2 = returns[1:, :]@alpha_2.values
port_ret_10 = returns[1:, :]@alpha_10.values
port_ret_2_d = returns[1:, :]*d_alpha_2
port_ret_2_d = port_ret_2_d.sum(axis=1)
port_ret_2_d = port_ret_2_d.reshape(port_ret_2_d.shape[0], 1)
port_ret_10_d = returns[1:, :]*d_alpha_10
port_ret_10_d = port_ret_10_d.sum(axis=1)
port_ret_10_d = port_ret_10_d.reshape(port_ret_10_d.shape[0], 1)

port_ret_2_cum = np.cumsum(np.log(1+port_ret_2), axis=0)
port_ret_10_cum = np.cumsum(np.log(1+port_ret_10), axis=0)
port_ret_2_d_cum = np.cumsum(np.log(1+port_ret_2_d), axis=0)
port_ret_10_d_cum = np.cumsum(np.log(1+port_ret_10_d), axis=0)

fig7 = plt.figure()
plt.plot(port_ret_2_cum)
plt.plot(port_ret_2_d_cum)
plt.plot(port_ret_10_cum)
plt.plot(port_ret_10_d_cum)
plt.legend(['static l=2', 'dynamic l=2', 'static l=10', 'dynamic l=10'])
plt.title('cumulative return')
plt.xlabel('Time')
plt.ylabel('returns')
fig7.savefig('fig7.png')
plt.show()

# Q4e
T = port_ret_10_d_cum.shape[0]
TC_2 = np.zeros((T, 1))
TC_10 = np.zeros((T, 1))

TC_2[0, 0] = np.abs(d_alpha_2[0, 0]) + np.abs(d_alpha_2[0, 1])
TC_2[1:, 0] = np.abs(d_alpha_2[1:, 0]-d_alpha_2[:-1, 0]) + np.abs(d_alpha_2[1:, 1]-d_alpha_2[:-1, 1])
TC_10[0, 0] = np.abs(d_alpha_10[0, 0]) + np.abs(d_alpha_10[0, 1])
TC_10[1:, 0] = np.abs(d_alpha_10[1:, 0]-d_alpha_10[:-1, 0]) + np.abs(d_alpha_10[1:, 1]-d_alpha_10[:-1, 1])

tao2 = (port_ret_2_d_cum[-1, 0]-port_ret_2_cum[-1, 0])/TC_2.sum()
print(tao2)
tao10 = (port_ret_10_d_cum[-1, 0]-port_ret_10_cum[-1, 0])/TC_10.sum()
print(tao10)
new_ret_2d = port_ret_2_d-TC_2*tao2
new_ret_10d = port_ret_10_d-TC_10*tao10
new_ret_2d_cum = np.cumsum(np.log(1+new_ret_2d))
new_ret_10d_cum = np.cumsum(np.log(1+new_ret_10d))

fig8 = plt.figure()
plt.plot(port_ret_2_cum)
plt.plot(new_ret_2d_cum)
plt.plot(port_ret_10_cum)
plt.plot(new_ret_10d_cum)
plt.legend(['static l=2', 'dynamic l=2', 'static l=10', 'dynamic l=10'])
plt.title('cumulative return')
plt.xlabel('Time')
plt.ylabel('returns')
fig8.savefig('fig8.png')
plt.show()
