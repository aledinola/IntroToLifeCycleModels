function varargout=ValueFnIter_Case1_FHorz_QuasiHyperbolic_e_raw(n_d,n_a,n_z,n_e,N_j, d_grid, a_grid, z_grid, e_grid,pi_z,pi_e, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
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
N_z=prod(n_z);
N_e=prod(n_e);

V=zeros(N_a,N_z,N_e,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_e,N_j,'gpuArray'); % indexes the optimal choice for d and aprime rest of dimensions a,z

%%
if length(DiscountFactorParamNames)<3
    error('There should be at least three variables in DiscountFactorParamNames when using Epstein-Zin Preferences')
end

eval('fieldexists_pi_z_J=1;vfoptions.pi_z_J;','fieldexists_pi_z_J=0;')
eval('fieldexists_ExogShockFn=1;vfoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;vfoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')

eval('fieldexists_EiidShockFn=1;vfoptions.EiidShockFn;','fieldexists_EiidShockFn=0;')
eval('fieldexists_EiidShockFnParamNames=1;vfoptions.EiidShockFnParamNames;','fieldexists_EiidShockFnParamNames=0;')
eval('fieldexists_pi_e_J=1;vfoptions.pi_e_J;','fieldexists_pi_e_J=0;')

if vfoptions.lowmemory>0
    special_n_e=ones(1,length(n_e));
    % e_gridvals is created below
end
if vfoptions.lowmemory>1
    l_z=length(n_z);
    special_n_z=ones(1,l_z);
    % z_gridvals is created below
end

%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);


if fieldexists_pi_z_J==1
    z_grid=vfoptions.z_grid_J(:,N_j);
    pi_z=vfoptions.pi_z_J(:,:,N_j);
elseif fieldexists_ExogShockFn==1
    if fieldexists_ExogShockFnParamNames==1
        ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,N_j);
        ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
        for ii=1:length(ExogShockFnParamsVec)
            ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
        end
        [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
        z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
    else
        [z_grid,pi_z]=vfoptions.ExogShockFn(N_j);
        z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
    end
end
if fieldexists_pi_e_J==1
    e_grid=vfoptions.e_grid_J(:,N_j);
    pi_e=vfoptions.pi_e_J(:,N_j);
elseif fieldexists_EiidShockFn==1
    if fieldexists_EiidShockFnParamNames==1
        EiidShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.EiidShockFnParamNames,N_j);
        EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
        for ii=1:length(EiidShockFnParamsVec)
            EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
        end
        [e_grid,pi_e]=vfoptions.EiidShockFn(EiidShockFnParamsCell{:});
        e_grid=gpuArray(e_grid); pi_e=gpuArray(pi_e);
    else
        [e_grid,pi_e]=vfoptions.ExogShockFn(N_j);
        e_grid=gpuArray(e_grid); pi_e=gpuArray(pi_e);
    end
end

pi_e=shiftdim(pi_e,-2); % Move to third dimension

if vfoptions.lowmemory>0
    if all(size(z_grid)==[sum(n_z),1])
        z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
    elseif all(size(z_grid)==[prod(n_z),l_z])
        z_gridvals=z_grid;
    end
    if all(size(e_grid)==[sum(n_e),1]) % kronecker (cross-product) grid
        e_gridvals=CreateGridvals(n_e,e_grid,1); % The 1 at end indicates want output in form of matrix.
    elseif all(size(e_grid)==[prod(n_e),length(n_e)]) % joint-grid
        e_gridvals=e_grid;
    end
end

if vfoptions.lowmemory==0
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_grid, e_grid, ReturnFnParamsVec);
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,:,:,N_j)=Vtemp;
    Policy(:,:,:,N_j)=maxindex;

elseif vfoptions.lowmemory==1
    
    for e_c=1:N_e
        e_val=e_gridvals(e_c,:);
        ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, special_n_e, d_grid, a_grid, z_grid, e_val, ReturnFnParamsVec);
        % Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix_e,[],1);
        V(:,:,e_c,N_j)=Vtemp;
        Policy(:,:,e_c,N_j)=maxindex;
    end
    
elseif vfoptions.lowmemory==2
    
    for e_c=1:N_e
        e_val=e_gridvals(e_c,:);
        for z_c=1:N_z
            z_val=z_gridvals(z_c,:);
            ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_grid, a_grid, z_val, e_val, ReturnFnParamsVec);
            % Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_ze,[],1);
            V(:,z_c,e_c,N_j)=Vtemp;
            Policy(:,z_c,e_c,N_j)=maxindex;
        end
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

    if fieldexists_pi_z_J==1
        z_grid=vfoptions.z_grid_J(:,jj);
        pi_z=vfoptions.pi_z_J(:,:,jj);
    elseif fieldexists_ExogShockFn==1
        if fieldexists_ExogShockFnParamNames==1
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
            ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
            for ii=1:length(ExogShockFnParamsVec)
                ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
            end
            [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
            z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
        else
            [z_grid,pi_z]=vfoptions.ExogShockFn(jj);
            z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
        end
    end
    if fieldexists_pi_e_J==1
        e_grid=vfoptions.e_grid_J(:,jj);
        pi_e=vfoptions.pi_e_J(:,jj);
        pi_e=shiftdim(pi_e,-2); % Move to thrid dimension
    elseif fieldexists_EiidShockFn==1
        if fieldexists_EiidShockFnParamNames==1
            EiidShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.EiidShockFnParamNames,jj);
            EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
            for ii=1:length(EiidShockFnParamsVec)
                EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
            end
            [e_grid,pi_e]=vfoptions.EiidShockFn(EiidShockFnParamsCell{:});
            e_grid=gpuArray(e_grid); pi_e=gpuArray(pi_e);
        else
            [e_grid,pi_e]=vfoptions.EiidShockFn(jj);
            e_grid=gpuArray(e_grid); pi_e=gpuArray(pi_e);
        end
        pi_e=shiftdim(pi_e,-2); % Move to third dimension
    end
    
    if vfoptions.lowmemory>0
        if (vfoptions.paroverz==1 || vfoptions.lowmemory==2) && (fieldexists_pi_z_J==1 || fieldexists_ExogShockFn==1)
            if all(size(z_grid)==[sum(n_z),1])
                z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
            elseif all(size(z_grid)==[prod(n_z),l_z])
                z_gridvals=z_grid;
            end
        end
        if (fieldexists_pi_e_J==1 || fieldexists_EiidShockFn==1)
            if all(size(e_grid)==[sum(n_e),1]) % kronecker (cross-product) grid
                e_gridvals=CreateGridvals(n_e,e_grid,1); % The 1 at end indicates want output in form of matrix.
            elseif all(size(e_grid)==[prod(n_e),length(n_e)]) % joint-grid
                e_gridvals=e_grid;
            end
        end
    end
    
    if strcmp(vfoptions.quasi_hyperbolic,'Naive')
        VKronNext_j=V(:,:,:,jj+1); % Use V (goes into the equation to determine V)
    else % strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
        VKronNext_j=Vunderbar(:,:,:,jj+1); % Use Vunderbar (goes into the equation to determine Vhat)
    end
            
    VKronNext_j=sum(VKronNext_j.*pi_e,3);
    
    if vfoptions.lowmemory==0
        
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_grid, e_grid, ReturnFnParamsVec);
        % (d,aprime,a,z,e)
        
        EV=VKronNext_j.*shiftdim(pi_z',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        entireEV=repelem(EV,N_d,1,1);
            
        if strcmp(vfoptions.quasi_hyperbolic,'Naive')
            % For naive, we compue V which is the exponential
            % discounter case, and then from this we get Vtilde and
            % Policy (which is Policytilde) that correspond to the
            % naive quasihyperbolic discounter
            % First V
            entireRHS=ReturnMatrix+beta*repmat(entireEV,1,N_a,1,N_e); % Use the two-future-periods discount factor
            [Vtemp,~]=max(entireRHS,[],1);
            V(:,:,:,jj)=shiftdim(Vtemp,1);
            % Now Vtilde and Policy
            entireRHS=ReturnMatrix+beta0beta*repmat(entireEV,1,N_a,1,N_e);
            [Vtemp,maxindex]=max(entireRHS,[],1);
            Vtilde(:,:,:,jj)=shiftdim(Vtemp,1); % Evaluate what would have done under exponential discounting
            Policy(:,:,:,jj)=shiftdim(maxindex,1); % Use the policy from solving the problem of Vtilde
        elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
            % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat)
            % and then we compute Vunderbar.
            % First Vhat
            entireRHS=ReturnMatrix+beta0beta*repmat(entireEV,1,N_a,1,N_e);  % Use the today-to-tomorrow discount factor
            [Vtemp,maxindex]=max(entireRHS,[],1);
            V(:,:,:,jj)=shiftdim(Vtemp,1); % Note that this is Vhat when sophisticated
            Policy(:,:,:,jj)=shiftdim(maxindex,1); % This is the policy from solving the problem of Vhat
            % Now Vstar
            entireRHS=ReturnMatrix+beta*repmat(entireEV,1,N_a,1,N_e); % Use the two-future-periods discount factor
            maxindexfull=maxindex+N_a*(0:1:N_a-1);
            Vunderbar(:,:,:,jj)=entireRHS(maxindexfull); % Evaluate time-inconsistent policy using two-future-periods discount rate
        end
        
    elseif vfoptions.lowmemory==1
        EV=VKronNext_j.*shiftdim(pi_z',-1);
        EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV=sum(EV,2); % sum over z', leaving a singular second dimension
        
        entireEV=repelem(EV,N_d,1,1);
        
        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            ReturnMatrix_e=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, special_n_e, d_grid, a_grid, z_grid, e_val, ReturnFnParamsVec);
                       
            if strcmp(vfoptions.quasi_hyperbolic,'Naive')
                % For naive, we compue V which is the exponential
                % discounter case, and then from this we get Vtilde and
                % Policy (which is Policytilde) that correspond to the
                % naive quasihyperbolic discounter
                % First V
                entireRHS_e=ReturnMatrix_e+beta*entireEV*ones(1,N_a,1); % Use the two-future-periods discount factor                
                [Vtemp,~]=max(entireRHS_e,[],1);
                V(:,:,e_c,jj)=shiftdim(Vtemp,1);
                % Now Vtilde and Policy
                entireRHS_e=ReturnMatrix_e+beta0beta*entireEV*ones(1,N_a,1);
                [Vtemp,maxindex]=max(entireRHS_e,[],1);
                Vtilde(:,:,e_c,jj)=shiftdim(Vtemp,1); % Evaluate what would have done under quasi-hyperbolic discounting
                Policy(:,:,e_c,jj)=shiftdim(maxindex,1); % Use the policy from solving the problem of Vtilde
            elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')  
                % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat) 
                % and then we compute Vunderbar.
                % First Vhat
                entireRHS_e=ReturnMatrix_e+beta0beta*entireEV*ones(1,N_a,1);  % Use the today-to-tomorrow discount factor
                [Vtemp,maxindex]=max(entireRHS_e,[],1);
                V(:,:,e_c,jj)=shiftdim(Vtemp,1); % Note that this is Vhat when sophisticated
                Policy(:,:,e_c,jj)=shiftdim(maxindex,1); % This is the policy from solving the problem of Vhat
                % Now Vstar
                entireRHS_e=ReturnMatrix_e+beta*entireEV*ones(1,N_a,1); % Use the two-future-periods discount factor
                maxindexfull=maxindex+N_a*(0:1:N_a-1);
                Vunderbar(:,:,e_c,jj)=entireRHS_e(maxindexfull); % Evaluate time-inconsistent policy using two-future-periods discount rate
            end
        end
        
    elseif vfoptions.lowmemory==2
        for e_c=1:N_e
            e_val=e_gridvals(e_c,:);
            for z_c=1:N_z
                z_val=z_gridvals(z_c,:);
                ReturnMatrix_ze=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, special_n_z, special_n_e, d_grid, a_grid, z_val, e_val, ReturnFnParamsVec);
                
                %Calc the condl expectation term (except beta), which depends on z but
                %not on control variables
                EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
                EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                EV_z=sum(EV_z,2);
                
                entireEV_z=kron(EV_z,ones(N_d,1));
                
                if strcmp(vfoptions.quasi_hyperbolic,'Naive')
                    % For naive, we compue V which is the exponential
                    % discounter case, and then from this we get Vtilde and
                    % Policy (which is Policytilde) that correspond to the
                    % naive quasihyperbolic discounter
                    % First V
                    entireRHS_ez=ReturnMatrix_ze+beta*entireEV_z*ones(1,N_a,1); % Use the two-future-periods discount factor
                    [Vtemp,~]=max(entireRHS_ez,[],1);
                    V(:,z_c,e_c,jj)=Vtemp;
                    % Now Vtilde and Policy
                    entireRHS_ez=ReturnMatrix_ze+beta0beta*entireEV_z*ones(1,N_a,1);
                    [Vtemp,maxindex]=max(entireRHS_ez,[],1);
                    Vtilde(:,z_c,e_c,jj)=Vtemp; % Evaluate what would have done under quasi-hyperbolic discounting
                    Policy(:,z_c,e_c,jj)=maxindex; % Use the policy from solving the problem of Vtilde
                elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
                    % For sophisticated we compute V, which is what we call Vhat, and the Policy (which is Policyhat)
                    % and then we compute Vunderbar.
                    % First Vhat
                    entireRHS_ez=ReturnMatrix_ze+beta0beta*entireEV_z*ones(1,N_a,1);  % Use the today-to-tomorrow discount factor
                    [Vtemp,maxindex]=max(entireRHS_ez,[],1);
                    V(:,z_c,e_c,jj)=Vtemp; % Note that this is Vhat when sophisticated
                    Policy(:,z_c,e_c,jj)=maxindex; % This is the policy from solving the problem of Vhat
                    % Now Vstar
                    entireRHS_ez=ReturnMatrix_ze+beta*entireEV_z*ones(1,N_a,1); % Use the two-future-periods discount factor
                    maxindexfull=maxindex+N_a*(0:1:N_a-1);
                    Vunderbar(:,z_c,e_c,jj)=entireRHS_ez(maxindexfull); % Evaluate time-inconsistent policy using two-future-periods discount rate
                end
            end
            
        end
        
    end
end

%%
Policy2=zeros(2,N_a,N_z,N_e,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
Policy2(1,:,:,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:,:,:)=shiftdim(ceil(Policy/N_d),-1);

if strcmp(vfoptions.quasi_hyperbolic,'Naive')
    varargout={Vtilde,Policy2}; % Policy will be Policytilde
else % strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
    varargout={V,Policy2}; % Policy will be Policyhat, V will be Vhat
end


end