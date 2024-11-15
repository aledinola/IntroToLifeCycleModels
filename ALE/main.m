%% Life-Cycle Model 20: Idiosyncratic shocks that depend on age
clear,clc,close all
%myf = 'C:\Users\aledi\Documents\GitHub\VFIToolkit-matlab';
myf = 'C:\Users\aledi\OneDrive\Documents\GitHub\VFIToolkit-matlab';
addpath(genpath(myf))

%% How does VFI Toolkit think about this?
% (aprime,a,z,theta,j)
% Endogenous state variable:           a, assets (total household savings)
% Stochastic exogenous state variable: z, 'unemployment' shock
% Permanent type:                      theta
% Age:                                 j

%% Begin setting up to use VFI Toolkit to solve
% Lets model agents from age 20 to age 100, so 81 periods

Params.agejshifter=19; % Age 20 minus one. Makes keeping track of actual age easy in terms of model age
Params.J=100-Params.agejshifter; % =81, Number of period in life-cycle

% Grid sizes to use
n_d = 0; % Endogenous labour choice (fraction of time worked)
n_a = 1000; % Endogenous asset holdings
n_z = 2; % Exogenous labor productivity units shock
N_j = Params.J; % Number of periods in finite horizon

%% Parameters

% Discount rate
Params.beta = 0.96;
% Preferences
Params.sigma = 2; % Coeff of relative risk aversion (curvature of consumption)
%Params.eta   = 1.5; % Curvature of leisure (This will end up being 1/Frisch elasty)
%Params.psi   = 10; % Weight on leisure

% Prices
Params.w=1; % Wage
Params.r=0.05; % Interest rate (0.05 is 5%)

% Demographics
Params.agej=1:1:Params.J; % Is a vector of all the agej: 1,2,3,...,J
Params.Jr=46;

% Pensions
Params.pension=0.3;

% Age-dependent labor productivity units
Params.kappa_j=[linspace(0.5,2,Params.Jr-15),linspace(2,1,14),zeros(1,Params.J-Params.Jr+1)];


Params.dj=[0.006879, 0.000463, 0.000307, 0.000220, 0.000184, 0.000172, 0.000160, 0.000149, 0.000133, 0.000114, 0.000100, 0.000105, 0.000143, 0.000221, 0.000329, 0.000449, 0.000563, 0.000667, 0.000753, 0.000823,...
    0.000894, 0.000962, 0.001005, 0.001016, 0.001003, 0.000983, 0.000967, 0.000960, 0.000970, 0.000994, 0.001027, 0.001065, 0.001115, 0.001154, 0.001209, 0.001271, 0.001351, 0.001460, 0.001603, 0.001769, 0.001943, 0.002120, 0.002311, 0.002520, 0.002747, 0.002989, 0.003242, 0.003512, 0.003803, 0.004118, 0.004464, 0.004837, 0.005217, 0.005591, 0.005963, 0.006346, 0.006768, 0.007261, 0.007866, 0.008596, 0.009473, 0.010450, 0.011456, 0.012407, 0.013320, 0.014299, 0.015323,...
    0.016558, 0.018029, 0.019723, 0.021607, 0.023723, 0.026143, 0.028892, 0.031988, 0.035476, 0.039238, 0.043382, 0.047941, 0.052953, 0.058457, 0.064494,...
    0.071107, 0.078342, 0.086244, 0.094861, 0.104242, 0.114432, 0.125479, 0.137427, 0.150317, 0.164187, 0.179066, 0.194979, 0.211941, 0.229957, 0.249020, 0.269112, 0.290198, 0.312231, 1.000000]; 
% dj covers Ages 0 to 100
Params.sj=1-Params.dj(21:101); % Conditional survival probabilities
Params.sj(end)=0; % In the present model the last period (j=J) value of sj is actually irrelevant

%% Permanent type
N_i = 2;
Params.theta = [0.5,1];
PTypeDistParamNames={'theta_dist'};
Params.theta_dist=[0.6,0.4]; % Must sum to one

%% Grids
% The ^3 means that there are more points near 0 and near 10. We know from
% theory that the value function will be more 'curved' near zero assets,
% and putting more points near curvature (where the derivative changes the most) increases accuracy of results.
a_grid=50*(linspace(0,1,n_a).^3)'; % The ^3 means most points are near zero, which is where the derivative of the value fn changes most.

d_grid=[];

%% z_grid and pi_z age-dependent
% The prob of being unemployed should decrease with age

z_grid_J = repmat([0,1]',1,N_j); %0=unemployed, 1=employed
pi_z_J = zeros(n_z,n_z,N_j);

pi_z_young=[0.5,0.5; 0.5,0.5]; % Probability of remaining employed is 0.5
pi_z_old=[0.1,0.9;0.5,0.5]; % Probability of remaining employed is 0.9

weight_young = ones(N_j,1);%(N_j-Params.agej'+1)/N_j;

for jj=1:N_j
    pi_z_J(:,:,jj) = weight_young(jj)*pi_z_young+(1-weight_young(jj))*pi_z_old;
end

% z_grid_J = [0,1]';
% pi_z_J = [0.5,0.5; 0.5,0.5];

[mean_z,~,~,statdist_z]=MarkovChainMoments(z_grid_J(:,1),pi_z_J(:,:,1));

%% Now, create the return function 
DiscountFactorParamNames={'beta','sj'};

% Notice we still use 'LifeCycleModel8_ReturnFn'
ReturnFn = @(aprime,a,z,theta,agej,Jr,kappa_j,w,r,pension,sigma) Mod_ReturnFn(aprime,a,z,theta,agej,Jr,kappa_j,w,r,pension,sigma);

%% Now solve the value function iteration problem, just to check that things are working before we go to General Equilbrium
disp('Test ValueFnIter')
vfoptions=struct(); % Just using the defaults.
vfoptions.verbose = 1;
vfoptions.divideandconquer = 1;
vfoptions.level1n = 11;
tic;
[V, Policy]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,N_j,N_i, d_grid, a_grid, z_grid_J, pi_z_J, ReturnFn, Params, DiscountFactorParamNames, vfoptions);
toc

%% Initial distribution of agents at birth (j=1)
% Before we plot the life-cycle profiles we have to define how agents are
% at age j=1. We will give them all zero assets.
jequaloneDist=zeros([n_a,n_z],'gpuArray'); % Put no households anywhere on grid
jequaloneDist(1,:) = statdist_z; % All agents start with zero assets, and the stationary dist over exogenous shocks
%jequaloneDist(1,1) = 1;

simoptions.verbose = 1;

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

mu_a_PT1 = sum(StationaryDist.ptype001,[2,3]);
mu_a_PT2 = sum(StationaryDist.ptype002,[2,3]);

%% FnsToEvaluate are how we say what we want to graph the life-cycles of

FnsToEvaluate.consumption = @(aprime,a,z,theta,agej,Jr,kappa_j,w,r,pension) Mod_Cons(aprime,a,z,theta,agej,Jr,kappa_j,w,r,pension);
FnsToEvaluate.income = @(aprime,a,z,theta,agej,Jr,kappa_j,w,pension) Mod_Income(aprime,a,z,theta,agej,Jr,kappa_j,w,pension); % w*kappa_j*z*h is the labor earnings (note: h will be zero when z is zero, so could just use w*kappa_j*h)
FnsToEvaluate.assets = @(aprime,a,z) a; % a is the current asset holdings
FnsToEvaluate.frac_unemp=@(aprime,a,z) (z==0); % indicator for z=0 (unemployment) 

%% Calculate the life-cycle profiles
AgeStats=LifeCycleProfiles_FHorz_Case1_PType(StationaryDist,Policy,FnsToEvaluate,Params,n_d,n_a,n_z,N_j,N_i,d_grid,a_grid,z_grid_J,simoptions);

vars = {'income','consumption'};
cv = struct();
for ii=1:numel(vars)
    var_name = vars{ii};
    cv.(var_name) = AgeStats.(var_name).StdDeviation./AgeStats.(var_name).Mean;
end

figure
plot(a_grid,mu_a_PT1)
hold on
plot(a_grid,mu_a_PT2)
legend('No college','College')
title('Stationary Distribution')

figure
plot(Params.agej,AgeStats.income.Mean)
hold on
plot(Params.agej,AgeStats.consumption.Mean)
legend('Income','Consumption')
title('Income and Consumption')

figure
plot(Params.agej,AgeStats.assets.ptype001.Mean)
hold on
plot(Params.agej,AgeStats.assets.Mean)
hold on
plot(Params.agej,AgeStats.assets.ptype002.Mean)
legend('No college','All','College')
title('Assets')

figure
plot(Params.agej,AgeStats.frac_unemp.Mean)
title('Share of unemployed (mean)')

figure
plot(Params.agej,cv.income)
hold on
plot(Params.agej,cv.consumption)
legend('Income','Consumption')
title('Coefficient of variation')