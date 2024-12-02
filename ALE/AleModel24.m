%% Life-Cycle Model 24: Using Permanent Type to model fixed-effects
clear,clc,close all
addpath(genpath('C:\Users\aledi\OneDrive\Documents\GitHub\VFIToolkit-matlab'))
% The exogenous process on labor efficiency units now uses an approach common in the literature:
% Labor efficiency units are a combination of four components:
% 1) kappa_j, a deterministic profile of age
% 2) z, a persistent markov shock
% 3) e, a transitory i.i.d. shock
% 4) alpha_i, a fixed effect
%
% All of these were already present in Life-Cycle Model 11, except the fixed-effect alpha_i
%
% We want to have five different possible values of alpha_i, and to do this we use the 'permanent type', PType, feature.
%
% We here use the easiest approach, we will create alpha_i as a vector with
% five values. And set N_i=5 (N_i is the number of permanent types). The
% codes will automatically realise that because alpha_i has N_i values it
% is different by permanent type.
%
% To compute the agent distribution we need to say how many agents are of
% each type. We denote this 'alphadist' and it is a vector of weights that
% sum to one (we need to put the name of this in PTypeDistParamNames, like
% we would for a discount factor in DiscountFactorParamNames).
%
% All the commands we run look slightly different as they are the PType
% versions of the commands until now. Permanent types are much more
% powerful than just fixed-effects, and later models will show more options.
%
% Model statistics, like the life-cycle profiles we calculate here, are
% reported both for each permanent type (that is to say, conditional on the
% permanent type), and 'grouped' across the permanent types. 

%% How does VFI Toolkit think about this?
%
% One decision variable: h, labour hours worked
% One endogenous state variable: a, assets (total household savings)
% Two stochastic exogenous state variables: z and e, persistent and transitory shocks to labor efficiency units, respectively
% Age: j
% Permanent types: i

%% Begin setting up to use VFI Toolkit to solve
% Lets model agents from age 20 to age 100, so 81 periods

Params.agejshifter=19; % Age 20 minus one. Makes keeping track of actual age easy in terms of model age
Params.J=100-Params.agejshifter; % =81, Number of period in life-cycle

% Grid sizes to use
n_d = 11;    % Endogenous labor choice (fraction of time worked)
n_a = 201;   % Endogenous asset holdings
n_z = [7,2]; % (1) labor prod shock, (2) health shock
n_e = 3;     % iid shock
N_i = 2;     % Permanent type of agents
N_j = Params.J; % Number of periods in finite horizon

%% The parameter that depends on the permanent type
% Fixed-effect (parameter that varies by permanent type)
Params.theta_i= [0.8,1.2]; % Roughly: increase earnings by 50%, 30%, 0, -30%, -50%
Params.theta_dist=[0.5,0.5]; % Must sum to one
PTypeDistParamNames={'theta_dist'};

%% Parameters

% Discount rate
Params.beta = 0.96;
% Preferences
Params.sigma = 2; % Coeff of relative risk aversion (curvature of consumption)
Params.eta = 1.5; % Curvature of leisure (This will end up being 1/Frisch elasty)
Params.psi = 10; % Weight on leisure

% Prices
Params.w=1; % Wage
Params.r=0.05; % Interest rate (0.05 is 5%)

% Demographics
Params.agej=1:1:Params.J; % Is a vector of all the agej: 1,2,3,...,J
Params.Jr=46;

% Pensions
Params.pen=0.3;

% Age-dependent labor productivity units
Params.kappa_j=[linspace(0.5,2,Params.Jr-15),linspace(2,1,14),zeros(1,Params.J-Params.Jr+1)];
% persistent AR(1) process on idiosyncratic labor productivity units
Params.rho_z=0.9;
Params.sigma_epsilon_z=0.02;
% transitiory iid normal process on idiosyncratic labor productivity units
Params.sigma_epsilon_e=0.2; % Implictly, rho_e=0

% Conditional death probabilities
Params.dj=[0.006879, 0.000463, 0.000307, 0.000220, 0.000184, 0.000172, 0.000160, 0.000149, 0.000133, 0.000114, 0.000100, 0.000105, 0.000143, 0.000221, 0.000329, 0.000449, 0.000563, 0.000667, 0.000753, 0.000823,...
    0.000894, 0.000962, 0.001005, 0.001016, 0.001003, 0.000983, 0.000967, 0.000960, 0.000970, 0.000994, 0.001027, 0.001065, 0.001115, 0.001154, 0.001209, 0.001271, 0.001351, 0.001460, 0.001603, 0.001769, 0.001943, 0.002120, 0.002311, 0.002520, 0.002747, 0.002989, 0.003242, 0.003512, 0.003803, 0.004118, 0.004464, 0.004837, 0.005217, 0.005591, 0.005963, 0.006346, 0.006768, 0.007261, 0.007866, 0.008596, 0.009473, 0.010450, 0.011456, 0.012407, 0.013320, 0.014299, 0.015323,...
    0.016558, 0.018029, 0.019723, 0.021607, 0.023723, 0.026143, 0.028892, 0.031988, 0.035476, 0.039238, 0.043382, 0.047941, 0.052953, 0.058457, 0.064494,...
    0.071107, 0.078342, 0.086244, 0.094861, 0.104242, 0.114432, 0.125479, 0.137427, 0.150317, 0.164187, 0.179066, 0.194979, 0.211941, 0.229957, 0.249020, 0.269112, 0.290198, 0.312231, 1.000000]; 
% dj covers Ages 0 to 100
Params.sj=1-Params.dj(21:101); % Conditional survival probabilities
Params.sj(end)=0; % In the present model the last period (j=J) value of sj is actually irrelevant

%% Grids
a_max = 100;
a_grid=a_max*(linspace(0,1,n_a).^3)'; % The ^3 means most points are near zero, which is where the derivative of the value fn changes most.

% First, the AR(1) process z1
tauchenopt.parallel=0;
[z_grid1,pi_z1]=discretizeAR1_Tauchen(0,Params.rho_z,Params.sigma_epsilon_z,n_z(1),3,tauchenopt);
z_grid1=exp(z_grid1); % Take exponential of the grid
[mean_z1,~,~,~]=MarkovChainMoments(z_grid1,pi_z1); % Calculate the mean of the grid so as can normalise it
z_grid1=z_grid1/mean_z1; % Normalise the grid on z (so that the mean of z is 1)

% Second, the health shock z2
Params.cutoff = 0.75;
z_grid2 = [0.5,1]'; %0=bad health, 1=good health
pi_z2 = [0.6, 0.4;
         0.2, 0.8];
aux = pi_z2^1000;
z2_stationary = aux(1,:)';
z2_stationary = z2_stationary/sum(z2_stationary);
health_dist = [0.05,0.95]; % Initial distribution of z2 at age 1
% Now the iid normal process e
[e_grid,pi_e]=discretizeAR1_Tauchen(0,0,Params.sigma_epsilon_e,n_e,3,tauchenopt);
e_grid=exp(e_grid); % Take exponential of the grid
pi_e=pi_e(1,:)'; % Because it is iid, the distribution is just the first row (all rows are identical). We use pi_e as a column vector for VFI Toolkit to handle iid variables.
mean_e=pi_e'*e_grid; % Because it is iid, pi_e is the stationary distribution (you could just use MarkovChainMoments(), I just wanted to demonstate a handy trick)
e_grid=e_grid/mean_e; % Normalise the grid on z (so that the mean of e is 1)
% To use e variables we have to put them into the vfoptions and simoptions
vfoptions.n_e=n_e;
vfoptions.e_grid=e_grid;
vfoptions.pi_e=pi_e;
simoptions.n_e=vfoptions.n_e;
simoptions.e_grid=vfoptions.e_grid;
simoptions.pi_e=vfoptions.pi_e;

% Grid for labour choice
h_grid=linspace(0,1,n_d)'; % Notice that it is imposing the 0<=h<=1 condition implicitly
% Switch into toolkit notation
d_grid=h_grid;

z_grid = [z_grid1;z_grid2]; % size is [n_z(1)+n_z(2),1]
pi_z   = kron(pi_z2,pi_z1); % kron in reverse order
%z_grid = z_grid1; 
%pi_z   = pi_z1; 

%% Now, create the return function 
DiscountFactorParamNames={'beta','sj'};

ReturnFn=@(h,aprime,a,z1,z2,e,theta_i,kappa_j,w,sigma,psi,eta,agej,Jr,pen,r)...
    AleModel24_ReturnFn(h,aprime,a,z1,z2,e,theta_i,kappa_j,w,sigma,psi,eta,agej,Jr,pen,r);

%% Now solve the value function iteration problem, just to check that things are working before we go to General Equilbrium
disp('Test ValueFnIter')
vfoptions.verbose=1;
vfoptions.divideandconquer=0;
vfoptions.level1n = 11;
tic;
vfoptions.verbose=1; % Just so we can see feedback on progress
[V, Policy]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,N_j,N_i, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, vfoptions);
toc

disp('Check size of V:')
disp(size(V.ptype001))
disp([n_a,n_z(1),n_z(2),n_e,N_j])
disp('Check size of Policy:')
disp(size(Policy.ptype001))
disp([length(n_d)+length(n_a),n_a,n_z(1),n_z(2),n_e,N_j])

%% Now, we want to graph Life-Cycle Profiles

%% Initial distribution of agents at birth (j=1)
% We have to define how agents are at age j=1. We will give them all zero assets.
jequaloneDist=zeros([n_a,n_z(1),n_z(2),n_e]); % Put no households anywhere on grid
% (a,z1,z2,e) given j=1, PT is defined elasewhere
jequaloneDist(1,floor((n_z(1)+1)/2),:,floor((n_e+1)/2))=z2_stationary; 
% All agents start with zero assets, the median value of labor productivity
% shock and from exogenous distribution of health state


%% We now compute the 'stationary distribution' of households
Params.mewj=ones(1,Params.J); % Marginal distribution of households over age
for jj=2:length(Params.mewj)
    Params.mewj(jj)=Params.sj(jj-1)*Params.mewj(jj-1);
end
Params.mewj=Params.mewj./sum(Params.mewj); % Normalize to one
AgeWeightsParamNames={'mewj'}; % So VFI Toolkit knows which parameter is the mass of agents of each age
StatDist=StationaryDist_Case1_FHorz_PType(jequaloneDist,AgeWeightsParamNames,PTypeDistParamNames,Policy,n_d,n_a,n_z,N_j,N_i,pi_z,Params,simoptions);
% Again, we will explain in a later model what the stationary distribution
% is, it is not important for our current goal of graphing the life-cycle profile

% Check that the distribution sums to one for each type
sum(StatDist.ptype001,'all')
sum(StatDist.ptype002,'all')

[mu,Policy_cpu] = reshape_VandPolicy(StatDist,Policy,n_a,n_z,n_e,N_i,N_j);
% (a,z1,e,theta_i,j)
mu_a = sum(mu,[2,3,4,5,6]);

% figure
% plot(a_grid,mu_a)
% title('Distribution of assets')

%% FnsToEvaluate are how we say what we want to graph the life-cycles of
% Like with return function, we have to include (h,aprime,a,z) as first
% inputs, then just any relevant parameters.
FnsToEvaluate.hours=@(h,aprime,a,z1,z2,e) h; % h is fraction of time worked
FnsToEvaluate.earnings=@(h,aprime,a,z1,z2,e,theta_i,kappa_j,w,pen) w*kappa_j*theta_i*z1*z2*e*h+pen; 
FnsToEvaluate.assets=@(h,aprime,a,z1,z2,e) a; % a is the current asset holdings
FnsToEvaluate.theta_i=@(h,aprime,a,z1,z2,e,theta_i) theta_i; % theta_i is the fixed effect
FnsToEvaluate.share_sick=@(h,aprime,a,z1,z2,e,cutoff) (z2<cutoff);

%--- Conditional restrictions. Must return either 0 or 1
condres.sick = @(h,aprime,a,z1,z2,e,cutoff) (z2<cutoff);
condres.healthy = @(h,aprime,a,z1,z2,e,cutoff) (z2>cutoff);
% Add additional field to simoptions
simoptions.conditionalrestrictions = condres;

%% Calculate the life-cycle profiles
disp('LifeCycleProfiles_FHorz_Case1_PType')
simoptions.whichstats = [1,1,0,0,0,0,0];
tic
AgeStats=LifeCycleProfiles_FHorz_Case1_PType(StatDist, Policy, FnsToEvaluate, Params,n_d,n_a,n_z,N_j,N_i,d_grid, a_grid, z_grid, simoptions);
toc

%% Note that if we want statistics for the distribution as a whole we could use 
disp('EvalFnOnAgentDist_AllStats_FHorz_Case1_PType')
tic
AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1_PType(StatDist, Policy, FnsToEvaluate, Params,n_d,n_a,n_z,N_j,N_i,d_grid, a_grid, z_grid, simoptions);
toc

%% Plot

figure
plot(1:1:Params.J,AgeStats.share_sick.Mean)

figure
plot(1:1:Params.J,AgeStats.sick.earnings.Mean)
hold on
plot(1:1:Params.J,AgeStats.earnings.Mean)
hold on
plot(1:1:Params.J,AgeStats.healthy.earnings.Mean)
hold off
title('Earnings by health')
legend('Sick','All','Healthy')

%% My checks

% Recover aggregate variables from conditional averages. 
% Suppose X is our variable of interest (e.g. consumption) and we have the
% conditional average by age. Given this, we want to recover the
% unconditional average. We can do this:
%   E[X] = sum_j \{ E[X|j]*prob(j) \}
% To check, compare to AllStats.X.Mean


ave1.earnings = AllStats.earnings.Mean;

% Conditional only on age
ave2.earnings = sum(AgeStats.earnings.Mean.*Params.mewj);

disp(ave1.earnings)
disp(ave2.earnings)

ave1_sick.earnings = AllStats.sick.earnings.Mean;
ave2_sick.earnings = sum(AgeStats.sick.earnings.Mean.*Params.mewj);

%% Last check: compare conditional earnings given health=sick and given age

income_sick1 = AgeStats.sick.earnings.Mean;

Values = zeros(size(mu));

for j=1:N_j
for theta_c=1:N_i
for e_c=1:n_e
for z2_c=1:n_z(2)
for z1_c=1:n_z(1)
for a_c=1:n_a
    h = d_grid(Policy_cpu(1,a_c,z1_c,z2_c,e_c,theta_c,j));
    aprime = a_grid(Policy_cpu(2,a_c,z1_c,z2_c,e_c,theta_c,j));
    a_val = a_grid(a_c);
    z1 = z_grid1(z1_c);
    z2 = z_grid2(z2_c);
    e = e_grid(e_c);
    theta = Params.theta_i(theta_c);
    Values(a_c,z1_c,z2_c,e_c,theta_c,j) = ...
        FnsToEvaluate.earnings(h,aprime,a_val,z1,z2,e,theta,Params.kappa_j(j),Params.w,Params.pen);
end
end
end
end
end
end

income_sick2 = zeros(1,N_j);
for j=1:N_j
    num = sum(Values(:,:,1,:,:,j).*mu(:,:,1,:,:,j),"all");
    den = sum(mu(:,:,1,:,:,j),"all");
    income_sick2(1,j) = num/den;
end

err = abs(income_sick2-income_sick1);

figure
plot(err')

