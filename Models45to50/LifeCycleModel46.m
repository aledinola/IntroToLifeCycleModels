%% Life-Cycle Model 46: GMM Estimation of a Life-Cycle Model, again
% Same model as Life-Cycle Model 45
% This time we include the parameters that determine exogenous shocks and
% initial agent distribution among the parameters to be estimated.
% We also show how to restrict parameters (to be positive, to be 0 to 1, to
% be A to B).
% To give better chance of identifying the additional parameters we target
% the fraction of time worked, as well as earnings.

% Changes from Life-Cycle Model 45 are:
% - around line 90, we modify the setup of the exogenous shocks to be a
%   function, as we want to estimate the parameters
% - around line 125, we set up jequaloneDist as a function, as we want to
%   estimate the parameters (importantly: covariance matrices should be
%   reparametrized for estimation, and we do this)
% - around line 240, we now try to estimate more parameters, and have
%   additional targets (so weighting matrix and covariance of data moments
%   matrices are also updated). We restrict some of the parameters (e.g.,
%   the standard deviation parameter is restricted to be positive, and the
%   autocorrelation is restricted to be 0 to 1).


%% How does VFI Toolkit think about this?
%
% One decision variable: h, labour hours worked
% One endogenous state variable: a, assets (total household savings)
% One stochastic exogenous state variable: z, an AR(1) process (in logs), idiosyncratic shock to labor productivity units
% Age: j

%% Begin setting up to use VFI Toolkit to solve
% Lets model agents from age 20 to age 100, so 81 periods

Params.agejshifter=19; % Age 20 minus one. Makes keeping track of actual age easy in terms of model age
Params.J=100-Params.agejshifter; % =81, Number of period in life-cycle

% Grid sizes to use
n_d=51; % Endogenous labour choice (fraction of time worked)
n_a=201; % Endogenous asset holdings
n_z=21; % Exogenous labor productivity units shock
N_j=Params.J; % Number of periods in finite horizon

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
Params.pension=0.3;

% Age-dependent labor productivity units
Params.kappa_j=[linspace(0.5,2,Params.Jr-15),linspace(2,1,14),zeros(1,Params.J-Params.Jr+1)];
% Exogenous shock process: AR1 on labor productivity units
Params.rho_z=0.9;
Params.sigma_epsilon_z=0.03;

% Conditional survival probabilities: sj is the probability of surviving to be age j+1, given alive at age j
% Most countries have calculations of these (as they are used by the government departments that oversee pensions)
% In fact I will here get data on the conditional death probabilities, and then survival is just 1-death.
% Here I just use them for the US, taken from "National Vital Statistics Report, volume 58, number 10, March 2010."
% I took them from first column (qx) of Table 1 (Total Population)
% Conditional death probabilities
Params.dj=[0.006879, 0.000463, 0.000307, 0.000220, 0.000184, 0.000172, 0.000160, 0.000149, 0.000133, 0.000114, 0.000100, 0.000105, 0.000143, 0.000221, 0.000329, 0.000449, 0.000563, 0.000667, 0.000753, 0.000823,...
    0.000894, 0.000962, 0.001005, 0.001016, 0.001003, 0.000983, 0.000967, 0.000960, 0.000970, 0.000994, 0.001027, 0.001065, 0.001115, 0.001154, 0.001209, 0.001271, 0.001351, 0.001460, 0.001603, 0.001769, 0.001943, 0.002120, 0.002311, 0.002520, 0.002747, 0.002989, 0.003242, 0.003512, 0.003803, 0.004118, 0.004464, 0.004837, 0.005217, 0.005591, 0.005963, 0.006346, 0.006768, 0.007261, 0.007866, 0.008596, 0.009473, 0.010450, 0.011456, 0.012407, 0.013320, 0.014299, 0.015323,...
    0.016558, 0.018029, 0.019723, 0.021607, 0.023723, 0.026143, 0.028892, 0.031988, 0.035476, 0.039238, 0.043382, 0.047941, 0.052953, 0.058457, 0.064494,...
    0.071107, 0.078342, 0.086244, 0.094861, 0.104242, 0.114432, 0.125479, 0.137427, 0.150317, 0.164187, 0.179066, 0.194979, 0.211941, 0.229957, 0.249020, 0.269112, 0.290198, 0.312231, 1.000000]; 
% dj covers Ages 0 to 100
Params.sj=1-Params.dj(21:101); % Conditional survival probabilities
Params.sj(end)=0; % In the present model the last period (j=J) value of sj is actually irrelevant

% Warm glow of bequest
Params.warmglow1=0.3; % (relative) importance of bequests
Params.warmglow2=3; % bliss point of bequests (essentially, the target amount)
Params.warmglow3=Params.sigma; % By using the same curvature as the utility of consumption it makes it much easier to guess appropraite parameter values for the warm glow


%% Grids
% The ^3 means that there are more points near 0 and near 10. We know from
% theory that the value function will be more 'curved' near zero assets,
% and putting more points near curvature (where the derivative changes the most) increases accuracy of results.
a_grid=10*(linspace(0,1,n_a).^3)'; % The ^3 means most points are near zero, which is where the derivative of the value fn changes most.

vfoptions.ExogShockFn=@(rho_z,sigma_epsilon_z,n_z) LifeCycleModel46_ExogShocks(rho_z,sigma_epsilon_z,n_z);
% Contents of LifeCycleModel47_ExogShocks are essentially just a copy-paste of what we had in Life-Cycle Model 46
% Note, codes will look for these inputs in Params, so we need to add
Params.n_z=n_z;
% We also need to put ExogShockFn into simoptions
simoptions.ExogShockFn=vfoptions.ExogShockFn;
% Some empty placeholders
z_grid=[];
pi_z=[];

% Grid for labour choice
h_grid=linspace(0,1,n_d)'; % Notice that it is imposing the 0<=h<=1 condition implicitly
% Switch into toolkit notation
d_grid=h_grid;

%% Now, create the return function 
DiscountFactorParamNames={'beta','sj'};

% Use 'LifeCycleModel45_ReturnFn' (it is just a renamed copy of the return fn used by Life-Cycle Model 9)
ReturnFn=@(h,aprime,a,z,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj) LifeCycleModel45_ReturnFn(h,aprime,a,z,w,sigma,psi,eta,agej,Jr,pension,r,kappa_j,warmglow1,warmglow2,warmglow3,beta,sj)

%% Now solve the value function iteration problem, just to check that things are working before we go to estimation
disp('Test ValueFnIter')
vfoptions.divideandconquer=1; % faster, requires problem is monotone
tic;
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
toc

%% Initial distribution of agents at birth (j=1)
% Define how agents are at age j=1. We will set up a joint-normal
% distribution on assets and exogenous shocks.

% Set initial assets to be joint-normal distribution with
Params.initassets_mean=1; % parameter for the mean of initial assets
Params.initassets_stddev=0.5; % parameter for the std dev of initial assets
Params.initz_mean=0; % parameter for the mean of initial z
Params.initz_stddev=Params.sigma_epsilon_z/(1-Params.rho_z); % parameter for the std dev of initial z
Params.initcov_az=0; % zero covariance between the two

InitialDistCovarMatrix=[Params.initassets_stddev^2, Params.initcov_az; Params.initcov_az, Params.initz_stddev^2];

% Trying to simply parametrize a covariance matrix is not a good approach to estimation
% So instead, we follow Acharkov & Hansen (2021)
% http://discourse.vfitoolkit.com/t/parametrize-a-covariance-matrix-using-archakov-hansen-2021/252
% and parametrize the correlation matrix
% [covariance matrices must be positive semi-definite, so you want to use a parametrization that is guarantees it 
% will be positive semi-definite, otherwise you just get lots of errors]

% Before we start, just double-check that it is symmetric and positive semidefinite (that it is a Covariance matrix)
% https://au.mathworks.com/help/matlab/math/determine-whether-matrix-is-positive-definite.html
try chol(InitialDistCovarMatrix)
    disp('Matrix is symmetric positive definite.') % good
catch ME
    error('Matrix is not symmetric positive definite') % bad, is not a covariance matrix
end

% We can use Matlab function cov2corr() to get the correlation matrix (and vector of standard deviations) from our covariance matrix.
[StdDevVector,CorrMatrix] = cov2corr(InitialDistCovarMatrix);
% Note: the diagonals of the correlation matrix are always ones by definition/construction

% The key contribution of AH2021 is how we can parametrize this correlation matrix as a vector.
% AH2021, page 1701 "[we] parametrize correlation matrices using the off-diagonal elements of logC [C is correlation matrix]"
logC=logm(CorrMatrix); % logC
temp=tril(logC,-1); % off-diagonal elements (as matrix)
AHcorrvector=temp(temp~=0); % drop the zeros (so now as column vector)
AHcorrvector=AHcorrvector'; % as a row vector
if isempty(AHcorrvector) % When covariance is zero, have to handle this specially
    AHcorrvector=0;
end

% We now have the vector of standard deviations, and a vector that
% parametrizes the correlation matrix. Combined these give us a vector that parametrizes the covariance matrix.
CoVarParametrization=[StdDevVector,AHcorrvector];

% This parametrization is what we send as inputs to
% LifeCycleModel47_InitialDist. And there we calculate the covariance
% matrix from these parameters.
Params.ah2021p1=CoVarParametrization(1);
Params.ah2021p2=CoVarParametrization(2);
Params.ah2021p3=CoVarParametrization(3);

jequaloneDist=@(a_grid,z_grid,n_a,n_z,initassets_mean,initz_mean,ah2021p1,ah2021p2,ah2021p3) LifeCycleModel46_InitialDist(a_grid,z_grid,n_a,n_z,initassets_mean,initz_mean,ah2021p1,ah2021p2,ah2021p3);
% to set up the initial agent dist as a function we must include the a_grid and z_grid in simoptions
simoptions.a_grid=a_grid;
% We have to create a copy of z_grid as it does not exist
[simoptions.z_grid,~]=simoptions.ExogShockFn(Params.rho_z,Params.sigma_epsilon_z,n_z);

%% We now compute the 'stationary distribution' of households
% Start with a mass of one at initial age, use the conditional survival
% probabilities sj to calculate the mass of those who survive to next
% period, repeat. Once done for all ages, normalize to one
Params.mewj=ones(1,Params.J); % Marginal distribution of households over age
for jj=2:length(Params.mewj)
    Params.mewj(jj)=Params.sj(jj-1)*Params.mewj(jj-1);
end
Params.mewj=Params.mewj./sum(Params.mewj); % Normalize to one
AgeWeightParamNames={'mewj'}; % So VFI Toolkit knows which parameter is the mass of agents of each age
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);
% Again, we will explain in a later model what the stationary distribution
% is, it is not important for our current goal of graphing the life-cycle profile

%% FnsToEvaluate are how we say what we want to graph the life-cycles of
% Like with return function, we have to include (h,aprime,a,z) as first
% inputs, then just any relevant parameters.
FnsToEvaluate.fractiontimeworked=@(h,aprime,a,z) h; % h is fraction of time worked
FnsToEvaluate.earnings=@(h,aprime,a,z,w,kappa_j) w*kappa_j*z*h; % w*kappa_j*z*h is the labor earnings (note: h will be zero when z is zero, so could just use w*kappa_j*h)
FnsToEvaluate.assets=@(h,aprime,a,z) a; % a is the current asset holdings

% notice that we have called these fractiontimeworked, earnings and assets

%% Calculate the life-cycle profiles
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);

% For example
% AgeConditionalStats.earnings.Mean
% There are things other than Mean, but in our current deterministic model
% in which all agents are born identical the rest are meaningless.

%% Plot the life cycle profiles of fraction-of-time-worked, earnings, and assets

figure(1)
subplot(3,1,1); plot(1:1:Params.J,AgeConditionalStats.fractiontimeworked.Mean)
title('Life Cycle Profile (pre-calibration): Fraction Time Worked (h)')
subplot(3,1,2); plot(1:1:Params.J,AgeConditionalStats.earnings.Mean)
title('Life Cycle Profile (pre-calibration): Labor Earnings (w kappa_j h)')
subplot(3,1,3); plot(1:1:Params.J,AgeConditionalStats.assets.Mean)
title('Life Cycle Profile (pre-calibration): Assets (a)')



%% Everything is working fine, time to turn to GMM Estimation of this model
% We will estimate three preference parameters.
% Preferences
% Params.sigma = 2; % Coeff of relative risk aversion (curvature of consumption)
% Params.eta = 1.5; % Curvature of leisure (This will end up being 1/Frisch elasty)
% Params.psi = 10; % Weight on leisure

% As targets, we will use the age-conditional mean earnings.
% Since these targets come from our model, we know that the true parameter
% values are sigma=2, eta=1.5, psi=10. 

% Obviously we will want to give a different initial guess for these parameters.
% The contents of Params (as passed as an input the the estimtion command below)
% are used as initial guesses for the parameters to be estimated, so we can
% just set some initial guesses as
Params.sigma=1.5;
Params.eta=1.1;
Params.psi=5;
Params.rho_z=0.8;
Params.sigma_epsilon_z=0.05;
Params.ah2021p1=0.4;
Params.ah2021p2=0.3;
Params.ah2021p3=0; % Note, this is the true value


%% GMM setup, three simple steps
% First, just name all the model parameters we want to estimate
EstimParamNames={'sigma','eta','psi','rho_z','sigma_epsilon_z','ah2021p1','ah2021p2','ah2021p3'};
% EstimParamNames gives the names of all the model parameters that will be
% estimated, these parameters must all appear in Params, and the values in
% Params will be used as the initial values.
% All other parameters in Params will remain fixed.

% Add some restrictions to the parameters.
%  - we restrict the correlation of the initial dist to being 0 to 1
estimoptions.constrain0to1={'ah2021p3'};
%  - we restrict sigma_epsilon_z and the standard deviations of the initial dist to being positive
estimoptions.constrainpositive={'sigma_epsilon_z','ah2021p1','ah2021p2'};
%  - we restrict the autocorrletation rho_z to be between 0 and 0.95 [if you use 0 to 1, then 1 causes errors in the quadrature process to discretize the AR(1)]
estimoptions.constrainAtoB={'rho_z'};
estimoptions.constrainAtoBlimits.rho_z=[0,0.95]; % 'constrainAtoBlimits' has to have the fieldnames that match the parameter names in constrainAtoB

% Second, we need to say which model statistics we want to target
% We can target any model stat generated by the AllStats, and LifeCycleProfiles commands
% We set up a structure containing all of our targets
TargetMoments.AgeConditionalStats.earnings.Mean=AgeConditionalStats.earnings.Mean;
TargetMoments.AgeConditionalStats.fractiontimeworked.Mean=AgeConditionalStats.fractiontimeworked.Mean;
% Note: When setting up TargetMoments there are some rules you must follow
% There are two options TargetMomements.AgeConditionalStats and TargetMoments.AllStats (you can use both). 
% Within these you must follow the structure that you get when you run the commands
% AgeConditionalStats=LifeCycleProfiles_FHorz_Case1()
% and
% AllStats=EvalFnOnAgentDist_AggVars_FHorz_Case1()

% Note, targeting the retirement earnings would be silly, as the parameters
% are irrelevant to them. So let's drop them from what we want to estimate.
% This is easy, just set them to NaN and they will be ignored
TargetMoments.AgeConditionalStats.earnings.Mean(Params.Jr:end)=NaN; % drop the retirement periods from the estimation
TargetMoments.AgeConditionalStats.fractiontimeworked.Mean(Params.Jr:end)=NaN; % drop the retirement periods from the estimation

% Third, we need a weighting matrix.
% We will just use the identity matrix, which is a silly choice, but easy.
% In Life-Cycle Model 47 we look at better ideas for how to choose the weighting matrix.
WeightingMatrix=eye(sum(~isnan([TargetMoments.AgeConditionalStats.earnings.Mean,TargetMoments.AgeConditionalStats.fractiontimeworked.Mean])));

%% To be able to compute the confidence intervals for the estimated parameters, there is one other important input we need
% The variance-covariance matrix of the GMM moment conditions, which here
% just simplifies to the variance-covariance matrix of the 'data' moments.
% We will see get this from data in Life-Cycle Model 47, for now, here is one I prepared earlier.
CovarMatrixDataMoments=diag([[0.001, 0.001, 0.001, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002,0.003, 0.003, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.005, 0.005, 0.005, 0.004, 0.004, 0.003, 0.003, 0.003, 0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001],...
    [0.0053, 0.0051, 0.0049, 0.0045, 0.0039, 0.0030, 0.0021, 0.0013, 0.0007, 0.0003, 0.0002, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0002, 0.0002, 0.0003, 0.0003, 0.0003, 0.0004, 0.0004, 0.0004, 0.0004, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005]]);

%% Done, now we just do the estimation
ParametrizeParamsFn=[]; % not something we are using in this example (will be used in Life-Cycle Model 47 and 49)

% We want a FnsToEvaluate which is just earnings (solely as this will be faster as it only includes what we actually need)
FnsToEvaluate_forGMM.earnings=FnsToEvaluate.earnings;
FnsToEvaluate_forGMM.fractiontimeworked=FnsToEvaluate.fractiontimeworked;

estimoptions.verbose=1; % give feedback
[EstimParams, EstimParamsConfInts, estsummary]=EstimateLifeCycleModel_MethodOfMoments(EstimParamNames,TargetMoments, WeightingMatrix, CovarMatrixDataMoments, n_d,n_a,n_z,N_j, d_grid,a_grid,z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, jequaloneDist, AgeWeightParamNames, ParametrizeParamsFn, FnsToEvaluate_forGMM, estimoptions, vfoptions, simoptions);
% EstimParams is the estimated parameter values
% EstimParamsConfInts are the 90% confidence intervals for the estimated parameter values
% estsummary is a structure containing various info on how the estimation went, plus some output useful for analysis

% Looking at
EstimParams
% we can see that they are essentially the same as the 'true' values which were
% sigma=2, eta=1.5, psi=10
% rho_z=0.9, sigma_epsilon_z=0.03
% ah2021p1=0.5, ah2021p2=0.3
% ah2021p3=0

% We don't actually care about ah2021p1,ah2021p2 and ah2021p3. They are just
% a parametrization of the covariance matrix that we could cleanly
% estimate. We do care about initassets_stddev and initz_stddev and initcov_az. 
% So we want to get the values and confidence intervals for these.
% First two are easy, because initassets_stddev=ah2021p1, and
% initz_stddev=ah2021p2, so they just have the same confidence intervals.
% initcov_az is a function of ah2021p3, so we just evaluate the function to get
% the point estimate and we could get the standard deviation
% of initcov_az from the standard deviation of ah2021p3 using the Delta method
% https://en.wikipedia.org/wiki/Delta_method

% We will discuss EstimParamsConfInts and estsummary in Life-Cycle Models 47 & 48.

% If you were to look at the confidence intervals you would see that some
% parameters, like rho_z, ah2021p2 and ah2021p3 are poorly identified (large
% confidence intervals). This is unsuprising given what these parameters are 
% and what our target moments are. In practice we should be thinking more
% carefully about what moments to target for the parameters we want to
% estimate.

%% Show some model output based on the estimated parameters
for pp=1:length(EstimParamNames)
    Params.(EstimParamNames{pp})=EstimParams.(EstimParamNames{pp});
end
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Params,simoptions);
AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Params,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);
% Since our estimated parameters were essentially identical to the 'true'
% parameter values, obviously the following is going to look the same as
% the previous figure
figure(2)
subplot(3,1,1); plot(1:1:Params.J,AgeConditionalStats.fractiontimeworked.Mean)
title('Life Cycle Profile: Fraction Time Worked (h)')
subplot(3,1,2); plot(1:1:Params.J,AgeConditionalStats.earnings.Mean)
title('Life Cycle Profile: Labor Earnings (w kappa_j h)')
subplot(3,1,3); plot(1:1:Params.J,AgeConditionalStats.assets.Mean)
title('Life Cycle Profile: Assets (a)')














