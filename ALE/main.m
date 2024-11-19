%% Life-Cycle Model with health shocks
% TODO: Add medical expenses if bad health and welfare transfers 

clear,clc,close all
myf = 'C:\Users\aledi\Documents\GitHub\VFIToolkit-matlab';
addpath(genpath(myf))

%% How does VFI Toolkit think about this?
% Ordering of variables: (aprime,a,z(1),z(2),theta,j)
% 1 Endogenous state variable:           a, assets 
% 2 Stochastic exogenous state variable: z, income shock and health shock
% 1 Permanent type (low vs high):        theta
%   Age:                                 j

%% Set options
vfoptions=struct(); % Just using the defaults.
vfoptions.verbose = 1;
vfoptions.divideandconquer = 0;
vfoptions.level1n = 11;

simoptions.verbose = 1;
simoptions.whichstats      = zeros(7,1);
simoptions.whichstats(1:3) = 1;

%% Begin setting up to use VFI Toolkit to solve

% Grid sizes to use
n_d = 0; % Endogenous labour choice (fraction of time worked)
n_a = 500; % Endogenous asset holdings
n_z = [7,2]; % 1)Exogenous labor productivity shock, 2)exogen health shock

%% Parameters

Params.agejshifter=19; % Age 20 minus one. Makes keeping track of actual age easy in terms of model age
Params.J=100-Params.agejshifter; % =81, Number of period in life-cycle
% Demographics
Params.agej=1:1:Params.J; % Is a vector of all the agej: 1,2,3,...,J
Params.Jr=45;
N_j = Params.J; % Number of periods in finite horizon

% Discount rate
Params.beta = 0.98;
% Preferences
Params.sigma = 2; % Coeff of relative risk aversion (curvature of consumption)
%Params.eta   = 1.5; % Curvature of leisure (This will end up being 1/Frisch elasty)
%Params.psi   = 10; % Weight on leisure

% Prices
Params.w=1; % Wage
Params.r=0.05; % Interest rate (0.05 is 5%)

% Age-dependent labor productivity units
Params.kappa_j = zeros(1,Params.J);
Params.kappa_j(1:Params.Jr-1) = [1.0000, 1.0719, 1.1438, 1.2158, 1.2842, 1.3527, ...
              1.4212, 1.4897, 1.5582, 1.6267, 1.6952, 1.7217, ...
              1.7438, 1.7748, 1.8014, 1.8279, 1.8545, 1.8810, ...
              1.9075, 1.9341, 1.9606, 1.9623, 1.9640, 1.9658, ...
              1.9675, 1.9692, 1.9709, 1.9726, 1.9743, 1.9760, ...
              1.9777, 1.9700, 1.9623, 1.9546, 1.9469, 1.9392, ...
              1.9315, 1.9238, 1.9161, 1.9084, 1.9007, 1.8354, ...
              1.7701, 1.7048];

% Pensions
Params.pension=0.5*Params.w*sum(Params.kappa_j)/(Params.Jr-1);

% Survival probability
Params.dj=[0.006879, 0.000463, 0.000307, 0.000220, 0.000184, 0.000172, 0.000160, 0.000149, 0.000133, 0.000114, 0.000100, 0.000105, 0.000143, 0.000221, 0.000329, 0.000449, 0.000563, 0.000667, 0.000753, 0.000823,...
    0.000894, 0.000962, 0.001005, 0.001016, 0.001003, 0.000983, 0.000967, 0.000960, 0.000970, 0.000994, 0.001027, 0.001065, 0.001115, 0.001154, 0.001209, 0.001271, 0.001351, 0.001460, 0.001603, 0.001769, 0.001943, 0.002120, 0.002311, 0.002520, 0.002747, 0.002989, 0.003242, 0.003512, 0.003803, 0.004118, 0.004464, 0.004837, 0.005217, 0.005591, 0.005963, 0.006346, 0.006768, 0.007261, 0.007866, 0.008596, 0.009473, 0.010450, 0.011456, 0.012407, 0.013320, 0.014299, 0.015323,...
    0.016558, 0.018029, 0.019723, 0.021607, 0.023723, 0.026143, 0.028892, 0.031988, 0.035476, 0.039238, 0.043382, 0.047941, 0.052953, 0.058457, 0.064494,...
    0.071107, 0.078342, 0.086244, 0.094861, 0.104242, 0.114432, 0.125479, 0.137427, 0.150317, 0.164187, 0.179066, 0.194979, 0.211941, 0.229957, 0.249020, 0.269112, 0.290198, 0.312231, 1.000000]; 
% dj covers Ages 0 to 100
Params.sj=1-Params.dj(21:101); % Conditional survival probabilities
Params.sj(end)=0; % In the present model the last period (j=J) value of sj is actually irrelevant

% Income penalty for bad health. It varies by permanent type. We assume
% that the low type loses more if bad health happens
Params.varrho.low  = exp(-0.2);
Params.varrho.high = exp(-0.1);

%% Permanent type PT
N_i = {'low','high'}; % No college vs college, or fixed income effect
sigma_theta = 0.245; % Variance of PT
% Impose the two moment conditions:
% 0.5*theta(1)+0.5*theta(2)=0 (mean zero) ==> theta(2)=-theta(1)
% 0.5*theta(1)^2+0.5*theta(2)^2=sigma_theta  ==> theta(1)=sqrt(sigma_theta)
theta_grid = [-sqrt(sigma_theta),sqrt(sigma_theta)];
Params.theta = exp(theta_grid);
PTypeDistParamNames={'theta_dist'};
Params.theta_dist=[0.5,0.5]; 

%% Grids
a_min = 0;
a_max = 400;
a_spacing = 3;
a_grid=a_min+(a_max-a_min)*(linspace(0,1,n_a).^a_spacing)'; 
d_grid=[];

%% z_grid and pi_z age-dependent
% z = [zn,zh] where zn=income shock, zh=health shock
Params.sigma_eps = sqrt(0.022); % Stdev of innovation to z
Params.rho       = 0.985; % Persistence
[z_grid_J,pi_z_J,zh_grid_J,pi_zh_J,dist_health] = create_shocks(n_z,N_j,Params);

% Own calculation
zh_prob.low = zeros(n_z(2),Params.J);
zh_prob.low(:,1) = dist_health;
zh_prob.high = zeros(n_z(2),Params.J);
zh_prob.high(:,1) = dist_health;
for jj=1:Params.J-1
    zh_prob.low(:,jj+1) = zh_prob.low(:,jj)'*pi_zh_J.low(:,:,jj);
    zh_prob.high(:,jj+1) = zh_prob.high(:,jj)'*pi_zh_J.high(:,:,jj);
end
figure
plot(Params.agej,zh_prob.low(1,:))
hold on
plot(Params.agej,zh_prob.high(1,:))
legend('Low \theta','high \theta')
title('Share of ppl with bad health, for each type \theta')
xlabel('Age, j')

%% Now, create the return function 
DiscountFactorParamNames={'beta','sj'};

% Notice we still use 'LifeCycleModel8_ReturnFn'
ReturnFn = @(aprime,a,zn,zh,theta,agej,Jr,kappa_j,varrho,w,r,pension,sigma) Mod_ReturnFn(aprime,a,zn,zh,theta,agej,Jr,kappa_j,varrho,w,r,pension,sigma);

%% Now solve the value function iteration problem, just to check that things are working before we go to General Equilbrium
disp('Test ValueFnIter')

tic;
[V, Policy]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,N_j,N_i, d_grid, a_grid, z_grid_J, pi_z_J, ReturnFn, Params, DiscountFactorParamNames, vfoptions);
toc

%% Initial distribution of agents at birth (j=1)
% Before we plot the life-cycle profiles we have to define how agents are
% at age j=1. We will give them all zero assets.
jequaloneDist=zeros([n_a,n_z(1),n_z(2)],'gpuArray'); % Put no households anywhere on grid
%jequaloneDist(1,:) = statdist_z; % All agents start with zero assets, and the stationary dist over exogenous shocks
jequaloneDist(1,floor((n_z(1)+1)/2),:) = dist_health;
%jequaloneDist(1,1) = 1;

Params.mewj=ones(1,Params.J); % Marginal distribution of households over age
for jj=2:length(Params.mewj)
    Params.mewj(jj)=Params.sj(jj-1)*Params.mewj(jj-1);
end
Params.mewj=Params.mewj./sum(Params.mewj); % Normalize to one
AgeWeightsParamNames={'mewj'}; % So VFI Toolkit knows which parameter is the mass of agents of each age

%% We now compute the 'stationary distribution' of households

StationaryDist=StationaryDist_Case1_FHorz_PType(jequaloneDist,AgeWeightsParamNames,PTypeDistParamNames,Policy,n_d,n_a,n_z,N_j,N_i,pi_z_J,Params,simoptions);
% Again, we will explain in a later model what the stationary distribution
% is, it is not important for our current goal of graphing the life-cycle profile

mu_a_PT1 = sum(StationaryDist.low,[2,3,4]);
mu_a_PT2 = sum(StationaryDist.high,[2,3,4]);

%% FnsToEvaluate and conditional restrictions

%--- Functions to be evaluatated for statistics, etc.
FnsToEvaluate.consumption = @(aprime,a,zn,zh,theta,agej,Jr,kappa_j,varrho,w,r,pension) Mod_Cons(aprime,a,zn,zh,theta,agej,Jr,kappa_j,varrho,w,r,pension);
FnsToEvaluate.income = @(aprime,a,zn,zh,theta,agej,Jr,kappa_j,varrho,w,pension) Mod_Income(aprime,a,zn,zh,theta,agej,Jr,kappa_j,varrho,w,pension); 
FnsToEvaluate.assets = @(aprime,a,zn,zh) a; % a is the current asset holdings
FnsToEvaluate.frac_badhealth=@(aprime,a,zn,zh) (zh==0); % indicator for z=0 (bad health) 

%--- Conditional restrictions. Must return either 0 or 1
conditionalrestrictions.sick = @(aprime,a,zn,zh) (zh==0);
conditionalrestrictions.healthy = @(aprime,a,zn,zh) (zh==1);
% Add additional field to simoptions
simoptions.conditionalrestrictions = conditionalrestrictions;

[ave] = MyCondStats(StationaryDist,Policy,n_z,z_grid_J,a_grid,n_a,N_i,N_j,Params,FnsToEvaluate);

%% Calculate the life-cycle profiles
AgeStats=LifeCycleProfiles_FHorz_Case1_PType(StationaryDist,Policy,FnsToEvaluate,Params,n_d,n_a,n_z,N_j,N_i,d_grid,a_grid,z_grid_J,simoptions);

vars = {'income','consumption'};
cv = struct();
for ii=1:numel(vars)
    var_name = vars{ii};
    cv.(var_name) = AgeStats.(var_name).StdDeviation./AgeStats.(var_name).Mean;
end

%% Calculate aggregate statistics
AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1_PType(StationaryDist,Policy,FnsToEvaluate,Params,n_d,n_a,n_z,N_j,N_i,d_grid,a_grid,z_grid_J,simoptions);

figure
plot(a_grid,mu_a_PT1)
hold on
plot(a_grid,mu_a_PT2)
legend('No college','College')
title('Stationary Distribution')

% Average income and consumption over the lifecycle
figure
plot(Params.agej,AgeStats.income.Mean)
hold on
plot(Params.agej,AgeStats.consumption.Mean)
legend('Income','Consumption')
xlabel('Age, j')
title('Income and Consumption, Average')

% Assets by permanent type
figure
plot(Params.agej,AgeStats.assets.low.Mean)
hold on
plot(Params.agej,AgeStats.assets.Mean)
hold on
plot(Params.agej,AgeStats.assets.high.Mean)
legend('No college','All','College')
xlabel('Age, j')
title('Assets')

figure
plot(Params.agej,AgeStats.frac_badhealth.low.Mean)
hold on
plot(Params.agej,AgeStats.frac_badhealth.high.Mean)
legend('Low \theta','high \theta')
title('Share of ppl with bad health, for each type \theta')
xlabel('Age, j')

figure
plot(Params.agej,cv.income)
hold on
plot(Params.agej,cv.consumption)
xlabel('Age, j')
legend('Income','Consumption')
title('Coefficient of variation')

% Consumption by health status (sick vs healthy) and by education type (low vs high)
figure
plot(Params.agej,AgeStats.sick.consumption.low.Mean,':','Linewidth',2)
hold on 
plot(Params.agej,AgeStats.healthy.consumption.low.Mean,"-.",'Linewidth',2)
hold on
plot(Params.agej,AgeStats.sick.consumption.high.Mean,"--",'Linewidth',2)
hold on 
plot(Params.agej,AgeStats.healthy.consumption.high.Mean,'Linewidth',2)
legend('Bad health, no college','Good health, no college','Bad health, college','Good health, college')
xlabel('Age, j')
title('Consumption')

% Assets by health status (sick vs healthy) and by education type (low vs high)
figure
plot(Params.agej,AgeStats.sick.assets.low.Mean,':','Linewidth',2)
hold on 
plot(Params.agej,AgeStats.healthy.assets.low.Mean,"-.",'Linewidth',2)
hold on
plot(Params.agej,AgeStats.sick.assets.high.Mean,"--",'Linewidth',2)
hold on 
plot(Params.agej,AgeStats.healthy.assets.high.Mean,'Linewidth',2)
legend('Bad health, no college','Good health, no college','Bad health, college','Good health, college')
xlabel('Age, j')
title('Assets')

% cons, given zh=zh(1) sick,theta(1),age
figure
plot(Params.agej,AgeStats.sick.consumption.low.Mean)
hold on
plot(Params.agej,squeeze(ave.consumption(1,1,:)))
title('Average consumption given health=0 and PT is low')