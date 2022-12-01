function varargout=ValueFnIter_Case1_FHorz_QuasiHyperbolic_noz_raw(n_d,n_a,N_j, d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% (last two entries of) DiscountFactorParamNames contains the names for the two parameters relating to
% Quasi-hyperbolic preferences.
% Let V_j be the standard (exponential discounting) solution to the value fn problem
% The 'Naive' quasi-hyperbolic solution takes current actions as if the
% future agent take actions as if having time-consistent (exponential discounting) preferences.
% V_naive_j= u_t+ beta_0 *E[V_{j+1}]
% The 'Sophisticated' quasi-hyperbolic solution takes into account the time-inconsistent behaviour of their future self.
% Let Vunderbar_j be the exponential discounting value fn of the time-inconsistent policy function (aka. the policy-greedy exponential discounting value function of the time-inconsistent policy function)
% V_sophisticated_j=u_t+beta_0*E[Vunderbar_{j+1}]
% See documentation for a fuller explanation of this.

N_d=prod(n_d);
N_a=prod(n_a);

V=zeros(N_a,N_j,'gpuArray');
Policy=zeros(N_a,N_j,'gpuArray'); % indexes the optimal choice for d and aprime rest of dimensions a,z

%%
if length(DiscountFactorParamNames)<3
    error('There should be at least three variables in DiscountFactorParamNames when using Epstein-Zin Preferences')
end

if vfoptions.lowmemory>0
    special_n_a=ones(1,length(n_a));
    a_gridvals=CreateGridvals(n_a,a_grid,1); % The 1 at end indicates want output in form of matrix.
end


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);
% Nothing extra to do for final period with quasi-hyperbolic preferences

if vfoptions.lowmemory==0
    
    %if vfoptions.returnmatrix==2 % GPU
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, n_d, n_a, d_grid, a_grid, ReturnFnParamsVec);
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,N_j)=Vtemp;
    Policy(:,N_j)=maxindex;
    
elseif vfoptions.lowmemory==1

    %if vfoptions.returnmatrix==2 % GPU
    for a_c=1:N_a
        a_val=a_gridvals(a_c,:);
        ReturnMatrix_a=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, n_d, special_n_a, d_grid, a_val, ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix_a);
        V(a_c,N_j)=Vtemp;
        Policy(a_c,N_j)=maxindex;
    end
    
end

if strcmp(vfoptions.quasi_hyperbolic,'Naive')
    Vtilde=V;
else % strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
    Vunderbar=V;
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;
    
    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',jj, N_j)
    end
    
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    if length(DiscountFactorParamsVec)>2
        DiscountFactorParamsVec=[prod(DiscountFactorParamsVec(1:end-1));DiscountFactorParamsVec(end)];
    end
    beta=prod(DiscountFactorParamsVec(1:end-1)); % Discount factor between any two future periods
    beta0beta=prod(DiscountFactorParamsVec); % Discount factor between today and tomorrow.

    
    if strcmp(vfoptions.quasi_hyperbolic,'Naive')
        VKronNext_j=V(:,jj+1); % Use V (goes into the equation to determine V)
    else % strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
        VKronNext_j=Vunderbar(:,jj+1); % Use Vunderbar (goes into the equation to determine Vhat)
    end
    
    if vfoptions.lowmemory==0
        
        %if vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, n_d, n_a, d_grid, a_grid, ReturnFnParamsVec);
        
        entireEV=kron(VKronNext_j,ones(N_d,1));
        
        if strcmp(vfoptions.quasi_hyperbolic,'Naive')
            % For naive, we compue V which is the exponential
            % discounter case, and then from this we get Vtilde and
            % Policy (which is Policytilde) that correspond to the
            % naive quasihyperbolic discounter
            % First V
            entireRHS=ReturnMatrix+beta*entireEV*ones(1,N_a,1); % Use the two-future-periods discount factor
            [Vtemp,~]=max(entireRHS,[],1);
            V(:,jj)=Vtemp;
            % Now Vtilde and Policy
            entireRHS=ReturnMatrix+beta0beta*entireEV*ones(1,N_a,1);
            [Vtemp,maxindex]=max(entireRHS,[],1);
            Vtilde(:,jj)=Vtemp; % Evaluate what would have done under exponential discounting
            Policy(:,jj)=maxindex; % Use the policy from solving the problem of Vtilde
        elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
            % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat)
            % and then we compute Vunderbar.
            % First Vhat
            entireRHS=ReturnMatrix+beta0beta*entireEV*ones(1,N_a,1);  % Use the today-to-tomorrow discount factor
            [Vtemp,maxindex]=max(entireRHS,[],1);
            V(:,jj)=Vtemp; % Note that this is Vhat when sophisticated
            Policy(:,jj)=maxindex; % This is the policy from solving the problem of Vhat
            % Now Vstar
            entireRHS=ReturnMatrix+beta*entireEV*ones(1,N_a,1); % Use the two-future-periods discount factor
            maxindexfull=maxindex+N_a*(0:1:N_a-1);
            Vunderbar(:,jj)=entireRHS(maxindexfull); % Evaluate time-inconsistent policy using two-future-periods discount rate
        end
        
    elseif vfoptions.lowmemory==1
        
        entireEV=kron(VKronNext_j,ones(N_d,1));
        
        if strcmp(vfoptions.quasi_hyperbolic,'Naive')
            % For naive, we compue V which is the exponential
            % discounter case, and then from this we get Vtilde and
            % Policy (which is Policytilde) that correspond to the
            % naive quasihyperbolic discounter
            % First V
            for a_c=1:N_z
                a_val=a_gridvals(a_c,:);
                ReturnMatrix_a=CreateReturnFnMatrix_Case1_Disc_noz_Par2(ReturnFn, n_d, special_n_a, d_grid, a_val, ReturnFnParamsVec);
                
                entireRHS_a=ReturnMatrix_a+beta*entireEV; % Use the two-future-periods discount factor
                [Vtemp,~]=max(entireRHS_a,[],1);
                V(a_c,jj)=Vtemp;
                % Now Vtilde and Policy
                entireRHS_a=ReturnMatrix_a+beta0beta*entireEV;
                [Vtemp,maxindex]=max(entireRHS_a,[],1);
                Vtilde(a_c,jj)=Vtemp; % Evaluate what would have done under exponential discounting
                Policy(a_c,jj)=maxindex; % Use the policy from solving the problem of Vtilde
            end
        elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
            % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat)
            % and then we compute Vunderbar.
            % First Vhat
            for a_c=1:N_a
                a_val=a_gridvals(a_c,:);
                ReturnMatrix_a=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, special_n_a, d_grid, a_val, ReturnFnParamsVec);
                
                entireRHS_a=ReturnMatrix_a+beta0beta*entireEV;  % Use the today-to-tomorrow discount factor
                [Vtemp,maxindex]=max(entireRHS_a,[],1);
                V(a_c,jj)=Vtemp; % Note that this is Vhat when sophisticated
                Policy(a_c,jj)=maxindex; % This is the policy from solving the problem of Vhat
                % Now Vstar
                entireRHS_a=ReturnMatrix_a+beta*entireEV; % Use the two-future-periods discount factor
                Vunderbar(a_c,jj)=entireRHS_a(maxindex); % Evaluate time-inconsistent policy using two-future-periods discount rate
            end
        end
        
    end
end

%%
Policy2=zeros(2,N_a,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
Policy2(1,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:)=shiftdim(ceil(Policy/N_d),-1);

if strcmp(vfoptions.quasi_hyperbolic,'Naive')
    varargout={Vtilde,Policy2}; % Policy will be Policytilde
else % strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
    varargout={V,Policy2}; % Policy will be Policyhat, V will be Vhat
end


end