%% Life-Cycle Model 24: Using Permanent Type to model fixed-effects
% Changes wrt original example of Robert: 
% Added a second Markov shock z2 (health), age-dependent
% Compute moments using conditional restrictions by health (z2=0 vs z2=1)
clear,clc,close all
myf = fullfile('..','..','VFIToolkit-matlab');
addpath(genpath(myf))

%% How does VFI Toolkit think about this?
%
% One decision variable: h, labour hours worked
% One endogenous state variable: a, assets (total household savings)
% Two exogenous state variables: z (z1 is income and z2 is health)
%   Note that the second z shock is age-dependent
% One iid shock e
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
Params.theta_i= [0.8,1.2]; % Theta has mean equal to 1
Params.theta_dist=[0.5,0.5]; % Must sum to one
PTypeDistParamNames={'theta_dist'};

%% Parameters

% Discount rate
Params.beta = 0.96;
% Preferences
Params.sigma = 2;   % Coeff of relative risk aversion (curvature of consumption)
Params.eta   = 1.5; % Curvature of leisure
Params.psi   = 10;  % Weight on leisure

% Prices
Params.w=1;    % Wage
Params.r=0.05; % Interest rate (0.05 is 5%)

% Demographics
Params.agej=1:1:Params.J; % Is a vector of all the agej: 1,2,3,...,J
Params.Jr=46;             % Retirement age

% Pensions
Params.pen=0.3; % No income heterogeneity after retirement

% Age-dependent labor productivity units
Params.kappa_j=[linspace(0.5,2,Params.Jr-15),linspace(2,1,14),zeros(1,Params.J-Params.Jr+1)];
% persistent AR(1) process on idiosyncratic labor productivity units
%   log(z1') = rho*log(z1) + epsilon_z', epsilon_z ~ N(0,sigma_epsilon_z^2)
Params.rho_z=0.9;
Params.sigma_epsilon_z=0.02;
% transitiory iid normal process on idiosyncratic labor productivity units
%   log(e) follows a Normal distrib. with mean 0 and stdev sigma_epsilon_e
Params.sigma_epsilon_e=0.2; % Implictly, rho_e=0

% Conditional death probabilities
Params.dj=[0.006879, 0.000463, 0.000307, 0.000220, 0.000184, 0.000172, 0.000160, 0.000149, 0.000133, 0.000114, 0.000100, 0.000105, 0.000143, 0.000221, 0.000329, 0.000449, 0.000563, 0.000667, 0.000753, 0.000823,...
    0.000894, 0.000962, 0.001005, 0.001016, 0.001003, 0.000983, 0.000967, 0.000960, 0.000970, 0.000994, 0.001027, 0.001065, 0.001115, 0.001154, 0.001209, 0.001271, 0.001351, 0.001460, 0.001603, 0.001769, 0.001943, 0.002120, 0.002311, 0.002520, 0.002747, 0.002989, 0.003242, 0.003512, 0.003803, 0.004118, 0.004464, 0.004837, 0.005217, 0.005591, 0.005963, 0.006346, 0.006768, 0.007261, 0.007866, 0.008596, 0.009473, 0.010450, 0.011456, 0.012407, 0.013320, 0.014299, 0.015323,...
    0.016558, 0.018029, 0.019723, 0.021607, 0.023723, 0.026143, 0.028892, 0.031988, 0.035476, 0.039238, 0.043382, 0.047941, 0.052953, 0.058457, 0.064494,...
    0.071107, 0.078342, 0.086244, 0.094861, 0.104242, 0.114432, 0.125479, 0.137427, 0.150317, 0.164187, 0.179066, 0.194979, 0.211941, 0.229957, 0.249020, 0.269112, 0.290198, 0.312231, 1.000000]; 
% dj covers Ages 0 to 100
Params.sj=1-Params.dj(21:101); % Conditional survival probabilities
% J is the last period so s(J), the prob of being alive in J+1 given that
% you are alive in J, is equal to 0 by construction.
Params.sj(end)=0; 

%% Assets and labor grids
a_max = 100;
a_grid=a_max*(linspace(0,1,n_a).^3)'; % The ^3 means most points are near zero, which is where the derivative of the value fn changes most.
% Grid for labour choice
h_grid=linspace(0,1,n_d)'; % Notice that it is imposing the 0<=h<=1 condition implicitly

%% AR(1) process z1
tauchenopt.parallel=0;
[z1_grid,pi_z1]=discretizeAR1_Tauchen(0,Params.rho_z,Params.sigma_epsilon_z,n_z(1),3,tauchenopt);
z1_grid=exp(z1_grid); % Take exponential of the grid
[mean_z1,~,~,~]=MarkovChainMoments(z1_grid,pi_z1); % Calculate the mean of the grid so as can normalise it
z1_grid=z1_grid/mean_z1; % Normalise the grid on z (so that the mean of z is 1)

% Second, the health shock z2, which is age-dependent
Params.cut = 0.5; % productivity cut if health is bad
z2_grid = [0,1]'; %0=bad health, 1=good health
p_bb_j = linspace(0.3,0.6,N_j); % prob from bad health to bad health
p_gb_j = linspace(0.0,0.2,N_j); % prob from good health to bad health
pi_z2_J = zeros(n_z(2),n_z(2),N_j);
for j=1:N_j
    prob_j = [p_bb_j(j),1-p_bb_j(j);
              p_gb_j(j),1-p_gb_j(j)];
    if any(abs(sum(prob_j,2)-1)>1e-10)
        fprintf('j = %d \n',j)
        error('prob_j does not sum to one!')
    end
    pi_z2_J(:,:,j) = prob_j;
end
check_markov_age(pi_z2_J,n_z(2),N_j);

% Initial distribution of z2 at age 1, potentially different from the
% stationary distribution of pi_z2
health_dist = [0.05,0.95];

%% Now the iid normal process e
[e_grid,pi_e]=discretizeAR1_Tauchen(0,0,Params.sigma_epsilon_e,n_e,3,tauchenopt);
e_grid=exp(e_grid);   % Take exponential of the grid
pi_e  =pi_e(1,:)';    % Because it is iid, the distribution is just the first row (all rows are identical). We use pi_e as a column vector for VFI Toolkit to handle iid variables.
mean_e=pi_e'*e_grid;  % Because it is iid, pi_e is the stationary distribution (you could just use MarkovChainMoments(), I just wanted to demonstate a handy trick)
e_grid=e_grid/mean_e; % Normalise the grid on z (so that the mean of e is 1)
% To use e variables we have to put them into the vfoptions and simoptions
vfoptions.n_e=n_e;
vfoptions.e_grid=e_grid;
vfoptions.pi_e=pi_e;
simoptions.n_e=vfoptions.n_e;
simoptions.e_grid=vfoptions.e_grid;
simoptions.pi_e=vfoptions.pi_e;

%% Switch into toolkit notation
d_grid=h_grid;

z_grid = [z1_grid;z2_grid]; % size is [n_z1+n_z2,1]
% pi_z2_J is [nz2,nz2,J], pi_z1 is [nz1,nz1]
pi_z1_J = repmat(pi_z1,[1,1,N_j]); % becomes [nz1,nz1,J]
pi_z = zeros(prod(n_z),prod(n_z),N_j); % [nz1*nz2,nz1*nz2,J]
for j=1:N_j
    pi_z(:,:,j) = kron(pi_z2_J(:,:,j),pi_z1_J(:,:,j)); % kron in reverse order
end

check_markov_age(pi_z,prod(n_z),N_j);

%% Now, create the return function 
DiscountFactorParamNames={'beta','sj'};

ReturnFn=@(h,aprime,a,z1,z2,e,theta_i,kappa_j,w,sigma,psi,eta,agej,Jr,pen,r,cut)...
    AleModel24_ReturnFn(h,aprime,a,z1,z2,e,theta_i,kappa_j,w,sigma,psi,eta,agej,Jr,pen,r,cut);

%% Now solve the value function iteration problem, just to check that things are working before we go to General Equilbrium
disp('Test ValueFnIter')
vfoptions.verbose=1;
vfoptions.divideandconquer=0;
vfoptions.level1n = 11;
tic;
vfoptions.verbose=1; % Just so we can see feedback on progress
[V, Policy]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,N_j,N_i, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, vfoptions);
time_vfi=toc;

disp('Check size of V:')
disp(size(V.ptype001))
disp([n_a,n_z(1),n_z(2),n_e,N_j])
disp('Check size of Policy:')
disp(size(Policy.ptype001))
disp([length(n_d)+length(n_a),n_a,n_z(1),n_z(2),n_e,N_j])

%% Initial distribution of agents at birth (j=1)
% We have to define how agents are at age j=1. We will give them all zero assets.
jequaloneDist=zeros([n_a,n_z(1),n_z(2),n_e]); % Put no households anywhere on grid
jequaloneDist(1,floor((n_z(1)+1)/2),:,floor((n_e+1)/2))=health_dist;
% All agents start with 
% a:  zero assets 
% z1: median value
% z2: non-degenerate distribution
% e:  median value
% ***theta: This is permanent type, distrib is fixed and set elsewhere

%% We now compute the 'stationary distribution' of households
Params.mewj=ones(1,Params.J); % Marginal distribution of households over age
for jj=2:length(Params.mewj)
    Params.mewj(jj)=Params.sj(jj-1)*Params.mewj(jj-1);
end
Params.mewj=Params.mewj./sum(Params.mewj); % Normalize to one
AgeWeightsParamNames={'mewj'}; % So VFI Toolkit knows which parameter is the mass of agents of each age

tic
StatDist=StationaryDist_Case1_FHorz_PType(jequaloneDist,AgeWeightsParamNames,PTypeDistParamNames,Policy,n_d,n_a,n_z,N_j,N_i,pi_z,Params,simoptions);
time_dist=toc;

% Check that the distribution sums to one for each type
sum(StatDist.ptype001,'all')
sum(StatDist.ptype002,'all')

% Convert distribution and policy functions from toolkit GPU to arrays on 
% the cpu, adding the PT as an extra dimension in the arrays (instead of a 
% structure as in the toolkit) 
[mu,Policy_cpu] = reshape_VandPolicy(StatDist,Policy,n_a,n_z,n_e,N_i,N_j);
% State variables: (a,z1,z2,e,theta_i,j)
mu_a = sum(mu,[2,3,4,5,6]);

% figure
% plot(a_grid,mu_a)
% title('Distribution of assets')

%% FnsToEvaluate are how we say what we want to graph the life-cycles of
% Like with return function, we have to include (h,aprime,a,z) as first
% inputs, then just any relevant parameters.
FnsToEvaluate.hours=@(h,aprime,a,z1,z2,e) h; % h is fraction of time worked
FnsToEvaluate.earnings=@(h,aprime,a,z1,z2,e,theta_i,kappa_j,w,agej,Jr,pen,cut)...
    fun_earnings(h,aprime,a,z1,z2,e,theta_i,kappa_j,w,agej,Jr,pen,cut);
FnsToEvaluate.assets=@(h,aprime,a,z1,z2,e) a; % a is the current asset holdings
FnsToEvaluate.theta_i=@(h,aprime,a,z1,z2,e,theta_i) theta_i; % theta_i is the fixed effect
FnsToEvaluate.share_sick=@(h,aprime,a,z1,z2,e) (z2==0); %share of sick people
FnsToEvaluate.agej=@(h,aprime,a,z1,z2,e,agej) agej; % Age j

%--- Conditional restrictions. Must return either 0 or 1
condres.sick = @(h,aprime,a,z1,z2,e) (z2==0);
condres.healthy = @(h,aprime,a,z1,z2,e) (z2==1);
% Add additional field to simoptions
simoptions.conditionalrestrictions = condres;

%% Calculate Lorenz curve of assets manually
Values_assets1 = repmat(a_grid,[1,n_z(1),n_z(2),n_e,N_i,N_j]);
Values_assets1 = Values_assets1(:);
StatDist1      = mu(:);

[relpop,relz] = mylorenz(Values_assets1,StatDist1);

figure
plot(relpop,relpop,relpop,relz)

%% Calculate the life-cycle profiles
disp('LifeCycleProfiles_FHorz_Case1_PType')
% Computation of Gini with cond restric takes a HUGE amount of time!
simoptions.whichstats = [1,1,1,1,0,0,0];
simoptions.agegroupings=1:10:N_j; % 5-year bins
age_vec = 1:10:N_j;
tic
AgeStats=LifeCycleProfiles_FHorz_Case1_PType(StatDist, Policy, FnsToEvaluate, Params,n_d,n_a,n_z,N_j,N_i,d_grid, a_grid, z_grid, simoptions);
time_age=toc;

%% Note that if we want statistics for the distribution as a whole we could use 
disp('EvalFnOnAgentDist_AllStats_FHorz_Case1_PType')
tic
AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1_PType(StatDist, Policy, FnsToEvaluate, Params,n_d,n_a,n_z,N_j,N_i,d_grid, a_grid, z_grid, simoptions);
time_all=toc;

AllStats.assets.Gini

%% Plot

% Share of sick hourseholds
figure
plot(age_vec,AgeStats.share_sick.Mean)
xlabel('Age, j')
title('Share of sick households in the pop.')

% Earnings
figure
plot(age_vec,AgeStats.sick.earnings.Mean)
hold on
plot(age_vec,AgeStats.earnings.Mean)
hold on
plot(age_vec,AgeStats.healthy.earnings.Mean)
hold off
title('Earnings by health status')
xlabel('Age, j')
legend('Sick','All','Healthy')

% Worked hours h
figure
plot(age_vec,AgeStats.sick.hours.Mean)
hold on
plot(age_vec,AgeStats.hours.Mean)
hold on
plot(age_vec,AgeStats.healthy.hours.Mean)
hold off
title('Worked hours by health status')
xlabel('Age, j')
legend('Sick','All','Healthy')

% Assets
figure
plot(age_vec,AgeStats.sick.assets.Mean)
hold on
plot(age_vec,AgeStats.assets.Mean)
hold on
plot(age_vec,AgeStats.healthy.assets.Mean)
hold off
title('Assets by health status')
xlabel('Age, j')
legend('Sick','All','Healthy')

% Inequality for assets, whole population
cv_assets = AgeStats.assets.StdDev./max(AgeStats.assets.Mean,1e-12);
% At the start of life-cycle, everyone has zero assets, so std(assets)=0
% and mean(assets)=0, hence cv(assets)=NaN, but economically it should be
% zero

figure
plot(age_vec,AgeStats.assets.Gini)
title('Gini of Assets by age')
xlabel('Age, j')

figure
plot(age_vec,cv_assets)
title('Coefficient of variation of Assets by age')
xlabel('Age, j')

%% My checks
ave_by_health.earnings = [AllStats.sick.earnings.Mean,...
                          AllStats.earnings.Mean,...
                          AllStats.healthy.earnings.Mean];
ave_by_health.hours = [AllStats.sick.hours.Mean,...
                          AllStats.hours.Mean,...
                          AllStats.healthy.hours.Mean];
ave_by_health.assets = [AllStats.sick.assets.Mean,...
                          AllStats.assets.Mean,...
                          AllStats.healthy.assets.Mean];
gini_by_health.assets=[AllStats.sick.assets.Gini,...
                        AllStats.assets.Gini,...
                        AllStats.healthy.assets.Gini];

disp('SUMMARY STATISTICS')
fprintf('Average earnings <sick>    = %f \n',ave_by_health.earnings(1))
fprintf('Average earnings <all>     = %f \n',ave_by_health.earnings(2))
fprintf('Average earnings <healthy> = %f \n',ave_by_health.earnings(3))
disp('---')
fprintf('Average hours <sick>    = %f \n',ave_by_health.hours(1))
fprintf('Average hours <all>     = %f \n',ave_by_health.hours(2))
fprintf('Average hours <healthy> = %f \n',ave_by_health.hours(3))
disp('---')
fprintf('Average assets <sick>    = %f \n',ave_by_health.assets(1))
fprintf('Average assets <all>     = %f \n',ave_by_health.assets(2))
fprintf('Average assets <healthy> = %f \n',ave_by_health.assets(3))
disp('---')
fprintf('Gini assets <sick>    = %f \n',gini_by_health.assets(1))
fprintf('Gini assets <all>     = %f \n',gini_by_health.assets(2))
fprintf('Gini assets <healthy> = %f \n',gini_by_health.assets(3))

%% Last check: compare conditional earnings given health=sick and given age

%income_sick1 = AgeStats.sick.earnings.Mean;

Values = zeros(size(mu));
Values_assets =  zeros(size(mu));
for j=1:N_j
for theta_c=1:N_i
for e_c=1:n_e
for z2_c=1:n_z(2)
for z1_c=1:n_z(1)
for a_c=1:n_a
    h = d_grid(Policy_cpu(1,a_c,z1_c,z2_c,e_c,theta_c,j));
    aprime = a_grid(Policy_cpu(2,a_c,z1_c,z2_c,e_c,theta_c,j));
    a_val = a_grid(a_c);
    z1 = z1_grid(z1_c);
    z2 = z2_grid(z2_c);
    e = e_grid(e_c);
    theta = Params.theta_i(theta_c);
    Values_assets(a_c,z1_c,z2_c,e_c,theta_c,j) = FnsToEvaluate.assets(h,aprime,a_val,z1,z2,e);
    Values(a_c,z1_c,z2_c,e_c,theta_c,j) = ...
        FnsToEvaluate.earnings(h,aprime,a_val,z1,z2,e,theta,Params.kappa_j(j),...
        Params.w,Params.agej(j),Params.Jr,Params.pen,Params.cut);
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

%err = abs(income_sick2-income_sick1);

%disp('Earnings by age, given sick: Error toolkit vs my code:')
%disp(max(err))

% Average earnings given sick, in the whole population
num = sum(Values(:,:,1,:,:,:).*mu(:,:,1,:,:,:),"all");
den = sum(mu(:,:,1,:,:,:),"all"); %divide by mass of sick people
ave_earnings_sick_cpu = num/den;

fprintf('Average earnings, given sick, toolkit = %f \n',AllStats.sick.earnings.Mean)
fprintf('Average earnings, given sick, CPU     = %f \n',ave_earnings_sick_cpu)

% Average assets given sick/healthy, whole population - AllStats.sick.assets.Mean
%   (a,z1,z2,e,theta_i,j)
num = sum(Values_assets(:,:,1,:,:,:).*mu(:,:,1,:,:,:),"all");
den = sum(mu(:,:,1,:,:,:),"all");
ave_assets_sick_cpu = num/den;
num = sum(Values_assets(:,:,2,:,:,:).*mu(:,:,2,:,:,:),"all");
den = sum(mu(:,:,2,:,:,:),"all");
ave_assets_healthy_cpu = num/den;

[ave_assets_sick_cpu,ave_assets_healthy_cpu]
[AllStats.sick.assets.Mean,AllStats.healthy.assets.Mean]

%% Another check
% Average assets given health=sick
xx1 = AllStats.sick.assets.Mean; 
xx2 = sum(AgeStats.sick.assets.Mean.*AgeStats.share_sick.Mean.*AgeStats.agej.Mean)/(sum(AgeStats.share_sick.Mean.*AgeStats.agej.Mean));
[xx1,xx2]

disp('RUNNING TIMES FOR SUBPARTS OF PROGRAM:')
fprintf('Time for VFI: %f \n',time_vfi)
fprintf('Time for Distr: %f \n',time_dist)
fprintf('Time for AgeStats: %f \n',time_age)
fprintf('Time for AllStats: %f \n',time_all)
