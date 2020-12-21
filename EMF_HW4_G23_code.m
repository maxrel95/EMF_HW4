%==========================================================================
% EMPIRICAL METHODS FOR FINANCE
% Homework 4 Modeling Volatility: GARCH Models
%
% Maxime Borel, Joe Habib, Yoann Joye & Axel Bandelier
% Group 23
% Date: May 2020
%==========================================================================
clear; 
clc; 
close all;

restoredefaultpath      % restore search path to default
addpath(genpath('/Users/maxime/Dropbox/MSc in Finance - 4.2/EMF/EMF_HW_4/EMF_HW4_Group23/Kevin Sheppard Toolbox'))
clear RESTOREDEFAULTPATH_EXECUTED

%import data
data = importdata('/Users/maxime/Dropbox/MSc in Finance - 4.2/EMF/EMF_HW_4/EMF_HW4_Group23/DATA_HW4.xlsx');
%import date
date = data.data.Feuil1(:,1);
date = datetime(date,'ConvertFrom','excel');
%% Question 1
Returns=diff(data.data.Feuil1(:,2:3))./data.data.Feuil1(1:end-1,2:3);
stock.return=Returns(:,1);
bond.return=Returns(:,2);
rf=(data.data.Feuil1(2:end,4)./100)./52;
Returns2=[stock.return,bond.return,rf];

stock.er=stock.return-rf;
bond.er=bond.return-rf;
ER=[stock.er,bond.er];
%% Question 2
lambda=[2,10];
mu=mean(Returns)';
avgrf=mean(rf);
sigma=cov(Returns);
e=ones(2,1);
alpha_2=(1/lambda(1)).*inv(sigma)*(mu-e.*avgrf);
alpha_2(3,1)=1-e'*alpha_2;
alpha_10=(1/lambda(2)).*inv(sigma)*(mu-e.*avgrf);
alpha_10(3,1)=1-e'*alpha_10;

result.weightq1.return=mat2dataset([alpha_2,alpha_10],'ObsNames',{'Wstock','Wbonds','Wrf'},...
    'VarNames',{'Lambda2','Lambda10'});
export(result.weightq1.return,'file','results/weightq1return.xls')

%excess return
mu_er=mean(ER)';
alpha_2_er=(1/lambda(1))*inv(sigma)*mu_er;
alpha_10_er=(1/lambda(2))*inv(sigma)*mu_er;
alpha_2_er(3,1)=1-e'*alpha_2_er;
alpha_10_er(3,1)=1-e'*alpha_10_er;

result.weightq1.er=mat2dataset([alpha_2_er,alpha_10_er],'ObsNames',{'Wstocker','Wbondser','Wrf'},...
    'VarNames',{'Lambda2','Lambda10'});
export(result.weightq1.er,'file','results/weightq1er.xls')
%% Question 3
%3a lilliefors & Ljung
%lilliefors
[stock.lill.dec,stock.lill.pval,stock.lill.kstat,stock.lill.cv]=lillietest(stock.er); % 95%conflvl
[bond.lill.dec,bond.lill.pval,bond.lill.kstat,bond.lill.cv]=lillietest(bond.er);
result.lillq3a=[stock.lill.kstat,bond.lill.kstat;stock.lill.cv,bond.lill.cv;stock.lill.pval,bond.lill.pval];
result.lillq3a=mat2dataset(result.lillq3a,'VarNames',{'Stocker','Bondser'},...
    'ObsNames',{'Lilliefors','cv','pValue'});
export(result.lillq3a,'file','results/lillq3a.xls')

%ljungbox
[stock.ljung.Q,stock.ljung.pval]=ljungbox(stock.er,4);
[bond.ljung.Q,bond.ljung.pval]=ljungbox(bond.er,4);
cv_ljung=chi2inv(0.95,4);
result.ljungq3a=[stock.ljung.Q,stock.ljung.pval,bond.ljung.Q,bond.ljung.pval];
result.ljungq3a=mat2dataset(result.ljungq3a,'VarNames',{'Ljungstocker','pValue1','Ljungbonder','pValue2'},...
    'ObsNames',{'1','2','3','4'});
export(result.ljungq3a,'file','results/ljungq3a.xls')

%3b AR(1)
%compute param stock get residual and coefficient
stock.ar1=fitlm(stock.return(1:end-1),stock.return(2:end));
stock.eps=table2array(stock.ar1.Residuals(:,1));
stock.ar1coeff=table2array(stock.ar1.Coefficients); %revenir sur export des coefficient et r2

%compute param bonds get residual and coefficient
bond.ar1=fitlm(bond.return(1:end-1),bond.return(2:end));
bond.eps=table2array(bond.ar1.Residuals(:,1));
bond.ar1coeff=table2array(bond.ar1.Coefficients);

result.arq3b=[stock.ar1coeff(1,[1,3]),bond.ar1coeff(1,[1,3]);...
    stock.ar1coeff(2,[1,3]),bond.ar1coeff(2,[1,3]);...
    stock.ar1.Rsquared.Ordinary,0,bond.ar1.Rsquared.Ordinary,0];
result.arq3b=mat2dataset(result.arq3b,'varnames',{'Stockparam','StocktStat',...
    'Bondparam','BondtStat'},'obsnames',{'phi0','phi1','r2'});
export(result.arq3b,'file','results/arq3b.xls')

%3c
stock.epssquared=stock.eps.^2; %get the squared eps
stock.arch.X=[stock.epssquared(4:end-1),stock.epssquared(3:end-2),... %create a maxtrix of regressor
    stock.epssquared(2:end-3),stock.epssquared(1:end-4)];

stock.arch.mdl=fitlm(stock.arch.X,stock.epssquared(5:end)); %run regression

stock.arch.lmtest=length(stock.eps)*stock.arch.mdl.Rsquared.Ordinary; %test
cv_lmtest=chi2inv(0.95,4);
stock.arch.pval=1-chi2cdf(stock.arch.lmtest,4);

bond.epssquared=bond.eps.^2; %do the same for bonds
bond.arch.X=[bond.epssquared(4:end-1),bond.epssquared(3:end-2),...
    bond.epssquared(2:end-3),bond.epssquared(1:end-4)];
bond.arch.mdl=fitlm(bond.arch.X,bond.epssquared(5:end));
bond.arch.lmtest=length(bond.eps)*bond.arch.mdl.Rsquared.Ordinary; 
bond.arch.pval=1-chi2cdf(bond.arch.lmtest,4);

[stock.arch.ljungQ,stock.arch.ljungp]=ljungbox(stock.epssquared,4); %alternative from lecture notes we can do 
[bond.arch.ljungQ,bond.arch.ljungp]=ljungbox(bond.epssquared,4); % a Ljung box test

result.archq3clmtest=[stock.arch.lmtest,bond.arch.lmtest;...
    cv_lmtest,cv_lmtest;stock.arch.pval,bond.arch.pval;...
    stock.arch.ljungQ(4,1),bond.arch.ljungQ(4,1);...
    stock.arch.ljungp(4,1),bond.arch.ljungp(4,1)];
result.archq3clmtest=mat2dataset(result.archq3clmtest,'VarNames',{'Stock','Bond'},...
    'ObsNames',{'LMtstat','CV','pValue','Qstat4','pValueQ'});
export(result.archq3clmtest,'file','results/archq3clmtest.xls')

% 3d
%estime GARCH
[stock.garch.para, ~,stock.garch.ht, ~, stock.garch.VCV] = tarch(stock.eps,1,0,1);
[bond.garch.para, ~,bond.garch.ht, ~, bond.garch.VCV] = tarch(bond.eps,1,0,1);

%test param + sum alpha beta
stock.garch.tstat=stock.garch.para./sqrt(diag(stock.garch.VCV));
bond.garch.tstat=bond.garch.para./sqrt(diag(bond.garch.VCV));
stock.garch.sum=stock.garch.para(2)+stock.garch.para(3);
bond.garch.sum=bond.garch.para(2)+bond.garch.para(3);

%create table to summarize
result.garchq3d=[stock.garch.para(1),stock.garch.tstat(1),bond.garch.para(1),bond.garch.tstat(1);...
    stock.garch.para(2),stock.garch.tstat(2),bond.garch.para(2),bond.garch.tstat(2);...
    stock.garch.para(3),stock.garch.tstat(3),bond.garch.para(3),bond.garch.tstat(3);...
    stock.garch.sum,0,bond.garch.sum,0,];

result.garchq3d=mat2dataset(result.garchq3d,'varnames',{'Stockpara','StocktStat',...
    'Bondpara','BondtStat'},'obsnames',{'omega','alpha','beta','sum alpha beta'});
export(result.garchq3d,'file','results/garchq3d.xls')

%3e
n=52;
stock.garch.fc52=zeros(n,1); %initialize table to store prevision
bond.garch.fc52=zeros(n,1);

stock.garch.fc52(1)=stock.garch.para(1)+... %compute first value it is easy to get
    stock.garch.para(2).*stock.eps(end).^2+stock.garch.para(3).*stock.garch.ht(end);
bond.garch.fc52(1)=bond.garch.para(1)+...
    bond.garch.para(2).*bond.eps(end).^2+bond.garch.para(3).*bond.garch.ht(end);

for i=2:n % prediction 
    stock.garch.fc52(i)=stock.garch.para(1)+stock.garch.sum.*stock.garch.fc52(i-1);
    bond.garch.fc52(i)=bond.garch.para(1)+bond.garch.sum.*bond.garch.fc52(i-1);
end

stock.garch.fcvol52=sqrt(stock.garch.fc52).*100; %work with sigma easier to understand
bond.garch.fcvol52=sqrt(bond.garch.fc52).*100;

f0=figure();
subplot(1,2,1)
plot(stock.garch.fcvol52)
title('Weekly volatility forecast for stock')
legend('Conditional vol','location','southoutside')
xlabel('Number of weeks forecasted')
ylabel('Weekly volatility forecast in %')

subplot(1,2,2)
plot(bond.garch.fcvol52)
hold on
title('Weekly volatility forecast for bond')
legend('Conditional vol','location','southoutside')
xlabel('Number of weeks forecasted')
ylabel('Weekly volatility forecast in %')
saveas(f0,'results/volaforecast52','png');
%%%%
n=250;
stock.garch.uncovar=stock.garch.para(1)/(1-stock.garch.sum); %get unconditional variance
bond.garch.uncovar=bond.garch.para(1)/(1-bond.garch.sum);

stock.garch.fc250=zeros(n,1); %initialize table to store prevision
bond.garch.fc250=zeros(n,1);

stock.garch.fc250(1)=stock.garch.para(1)+... %compute first value using known data
    stock.garch.para(2).*stock.eps(end).^2+stock.garch.para(3).*stock.garch.ht(end);
bond.garch.fc250(1)=bond.garch.para(1)+...
    bond.garch.para(2).*bond.eps(end).^2+bond.garch.para(3).*bond.garch.ht(end);

for i=2:n % prediction 
    stock.garch.fc250(i)=stock.garch.para(1)+stock.garch.sum.*stock.garch.fc250(i-1);
    bond.garch.fc250(i)=bond.garch.para(1)+bond.garch.sum.*bond.garch.fc250(i-1);
end

stock.garch.fcvol250=sqrt(stock.garch.fc250).*100; %work with sigma easier to understand
bond.garch.fcvol250=sqrt(bond.garch.fc250).*100;

f1=figure();
subplot(1,2,1)
plot(stock.garch.fcvol250)
hold on
plot(sqrt(stock.garch.uncovar)*100.*ones(n,1))
hold off
title('Weekly volatility forecast for stock')
legend('Conditional vol','unconditional vol','location','southoutside')
xlabel('Number of weeks forecasted')
ylabel('Weekly volatility forecast in %')

subplot(1,2,2)
plot(bond.garch.fcvol250)
hold on
plot(sqrt(bond.garch.uncovar)*100.*ones(n,1))
hold off
title('Weekly volatility forecast for bond')
legend('Conditional vol','unconditional vol','location','southoutside')
xlabel('Number of weeks forecasted')
ylabel('Weekly volatility forecast in %')
saveas(f1,'results/volaforecast250','png');
%% Question 4
%4a
%we dont need to compute the sigmat because it is an output of tarch fun. 
%we dont need to compute  mut because it is already done by fitlm with
%fitted value 
%recal
stock.am.ht=stock.garch.ht;
stock.am.mut=stock.ar1.Fitted;
bond.am.ht=bond.garch.ht;
bond.am.mut=bond.ar1.Fitted;
T=length(stock.am.ht);

corrmat=corr(stock.eps,bond.eps);
covarsbt=corrmat .*sqrt(stock.am.ht) .*sqrt(bond.am.ht);
sigmat=zeros(2,2,T);
sigmat(1,1,:)=stock.am.ht;
sigmat(2,2,:)=bond.am.ht;
sigmat(1,2,:)=covarsbt;
sigmat(2,1,:)=covarsbt;

%4b
ptf.weight.l2=zeros(3,T); %store the weights 
ptf.weight.l10=zeros(3,T);

ptf.mut=[stock.am.mut';bond.am.mut']; %colum vector for each t of mut

for i=1:T
    ptf.weight.l2(1:2,i)=(1/lambda(1))*inv(sigmat(:,:,i))*(ptf.mut(:,i)-e.*rf(i+1)); %compute weight
    ptf.weight.l10(1:2,i)=(1/lambda(2))*inv(sigmat(:,:,i))*(ptf.mut(:,i)-e.*rf(i+1));
end

ptf.weight.l2(3,:)=1-sum(ptf.weight.l2(1:2,:));
ptf.weight.l10(3,:)=1-sum(ptf.weight.l10(1:2,:));

%4c
f2=figure();
plot(date(3:end),ptf.weight.l2')
hold on
plot(date(3:end),alpha_2'.*ones(T,3))
hold off
legend('Dynamic stock ','Dynamic bond ','Dynamic Rf',...
    'Static stock','Static bond','Static Rf','location','eastout')
title('Dynamic vs static allocation with lambda=2')
xlabel('Date')
ylabel('Weights')
saveas(f2,'results/dynamicallocationl2','png')

f3=figure();
plot(date(3:end),ptf.weight.l10')
hold on
plot(date(3:end),alpha_10'.*ones(T,3))
hold off
legend('Dynamic stock ','Dynamic bond ','Dynamic Rf',...
    'Static stock','Static bond','Static Rf','location','eastout')
title('Dynamic vs static allocation with lambda=10')
xlabel('Date')
ylabel('Weights')
saveas(f3,'results/dynamicallocationl10','png')

%4d
ptf.return.dl2=sum((ptf.weight.l2'.*Returns2(2:end,:)),2);
ptf.return.dl10=sum(ptf.weight.l10'.*Returns2(2:end,:),2);
ptf.return.sl2=sum((alpha_2'.*ones(T,3)).*Returns2(2:end,:),2);
ptf.return.sl10=sum((alpha_10'.*ones(T,3)).*Returns2(2:end,:),2);

ptf.scr.dl2=cumprod((1+ptf.return.dl2));
ptf.scr.dl10=cumprod((1+ptf.return.dl10));
ptf.scr.sl2=cumprod((1+ptf.return.sl2));
ptf.scr.sl10=cumprod((1+ptf.return.sl10));

ptf.lcr.dl2=cumsum(log(1+ptf.return.dl2));
ptf.lcr.dl10=cumsum(log(1+ptf.return.dl10));
ptf.lcr.sl2=cumsum(log(1+ptf.return.sl2));
ptf.lcr.sl10=cumsum(log(1+ptf.return.sl10));

f4=figure();
plot(date(3:end),ptf.lcr.dl2)
hold on
plot(date(3:end),ptf.lcr.dl10)
hold on
plot(date(3:end),ptf.lcr.sl2)
hold on
plot(date(3:end),ptf.lcr.sl10)
hold off
legend('Dynamic lambda=2','Dynamic lambda=10','Static lambda=2','Static lambda=10',...
    'location','bestoutside')
xlabel('Date')
ylabel('Cumulative portfolio return')
title('Cumulative portfolios log returns')
saveas(f4,'results/ptfcumlogreturn','png')

ptf.return.all=[ptf.return.dl2,ptf.return.dl10,ptf.return.sl2,ptf.return.sl10];
ptf.stat.mu=mean(ptf.return.all)*52;
ptf.stat.std=std(ptf.return.all)*sqrt(52);
ptf.stat.sr=(ptf.stat.mu-(mean(rf(3:end)))*52)./ptf.stat.std;
ptf.stat.skew=skewness(ptf.return.all);
ptf.stat.ekurt=kurtosis(ptf.return.all)-3;

result.q4d=[ptf.stat.mu;ptf.stat.std;ptf.stat.sr;...
    ptf.stat.skew;ptf.stat.ekurt];
result.q4d=mat2dataset(result.q4d,'varnames',{'dynl2','dynl10',...
    'staticl2','staticl10'},'obsnames',{'Annualized mu','annualized std',...
    'Sharperatio','Skew','ExcessKurt'});
export(result.q4d,'file','results/sumstatportfolio.xls')

%4e
TCl2=zeros(1,T);
TCl10=zeros(1,T);
TCl2(1)=abs(ptf.weight.l2(1,1))+abs(ptf.weight.l2(2,1));
TCl2(2:end)=abs(ptf.weight.l2(1,2:end)-ptf.weight.l2(1,1:end-1))+...
    abs(ptf.weight.l2(2,2:end)-ptf.weight.l2(2,1:end-1));
TCl10(1)=abs(ptf.weight.l10(1,1))+abs(ptf.weight.l10(2,1));
TCl10(2:end)=abs(ptf.weight.l10(1,2:end)-ptf.weight.l10(1,1:end-1))+...
    abs(ptf.weight.l10(2,2:end)-ptf.weight.l10(2,1:end-1));

taol2=(ptf.lcr.dl2(end)-ptf.lcr.sl2(end))/sum(TCl2);
taol10=(ptf.lcr.dl10(end)-ptf.lcr.sl10(end))/sum(TCl10);

ptf.returnTC.dl2=ptf.return.dl2-TCl2'.*taol2;
ptf.returnTC.dl10=ptf.return.dl10-TCl10'.*taol10;
ptf.lcrTC.dl2=cumsum(log(1+ptf.returnTC.dl2));
ptf.lcrTC.dl10=cumsum(log(1+ptf.returnTC.dl10));

f5=figure();
plot(date(3:end),ptf.lcrTC.dl2)
hold on
plot(date(3:end),ptf.lcrTC.dl10)
hold on
plot(date(3:end),ptf.lcr.sl2)
hold on
plot(date(3:end),ptf.lcr.sl10)
hold off
legend('Dynamic lambda=2','Dynamic lambda=10','Static lambda=2','Static lambda=10',...
    'location','bestoutside')
xlabel('Date')
ylabel('Cumulative portfolio return')
title('Cumulative portfolios log returns with transaction cost')
saveas(f5,'results/ptfcumlogreturnTC','png')
