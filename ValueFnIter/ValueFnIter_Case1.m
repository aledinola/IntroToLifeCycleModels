function varargout=ValueFnIter_Case1(n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Solves infinite-horizon 'Case 1' value function problems.
% Typically, varargoutput={V,Policy};

V=nan; % Matlab was complaining that V was not assigned

%% Check which vfoptions have been used, set all others to defaults 
if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
    vfoptions.solnmethod='purediscretization_refinement';
    vfoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    if vfoptions.parallel==2
        vfoptions.returnmatrix=2; % On GPU, must use this option
%         vfoptions.solnmethod='purediscretization_relativeVFI'; % Has only been implemented on the GPU
    end
    if isfield(vfoptions,'returnmatrix')==0
        if isa(ReturnFn,'function_handle')==1
            vfoptions.returnmatrix=0;
        else
            vfoptions.returnmatrix=1;
        end
    end
    vfoptions.lowmemory=0;
    vfoptions.verbose=0;
    vfoptions.tolerance=10^(-9);
    vfoptions.howards=80;
    vfoptions.maxhowards=500;
    vfoptions.endogenousexit=0;
    vfoptions.endotype=0; % (vector indicating endogenous state is a type)
    vfoptions.incrementaltype=0; % (vector indicating endogenous state is an incremental endogenous state variable)
%     vfoptions.exoticpreferences % default is not to declare it
%     vfoptions.SemiEndogShockFn % default is not to declare it    
    vfoptions.polindorval=1;
    vfoptions.policy_forceintegertype=0;
    vfoptions.piz_strictonrowsaddingtoone=0;
    vfoptions.outputkron=0;
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if isfield(vfoptions,'solnmethod')==0
        vfoptions.solnmethod='purediscretization_refinement';
    end
    if isfield(vfoptions,'parallel')==0
        vfoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    end
    if vfoptions.parallel==2
        vfoptions.returnmatrix=2; % On GPU, must use this option
    end
    if isfield(vfoptions,'returnmatrix')==0
        if isa(ReturnFn,'function_handle')==1
            vfoptions.returnmatrix=0;
        else
            vfoptions.returnmatrix=1;
        end
    end
    if isfield(vfoptions,'lowmemory')==0
        vfoptions.lowmemory=0;
    end
    if isfield(vfoptions,'verbose')==0
        vfoptions.verbose=0;
    end
     if isfield(vfoptions,'tolerance')==0
        vfoptions.tolerance=10^(-9);
    end
    if isfield(vfoptions,'howards')==0
        vfoptions.howards=80;
    end  
    if isfield(vfoptions,'maxhowards')==0
        vfoptions.maxhowards=500;
    end  
    if isfield(vfoptions,'endogenousexit')==0
        vfoptions.endogenousexit=0;
    end
    if isfield(vfoptions,'endotype')==0
        vfoptions.endotype=0; % (vector indicating endogenous state is a type)
    end
    if isfield(vfoptions,'incrementaltype')==0
        vfoptions.incrementaltype=0; % (vector indicating endogenous state is an incremental endogenous state variable)
    end
%     vfoptions.exoticpreferences % default is not to declare it
%     vfoptions.SemiEndogShockFn % default is not to declare it    
    if isfield(vfoptions,'polindorval')==0
        vfoptions.polindorval=1;
    end
    if isfield(vfoptions,'policy_forceintegertype')==0
        vfoptions.policy_forceintegertype=0;
    end
    if isfield(vfoptions,'piz_strictonrowsaddingtoone')==0
        vfoptions.piz_strictonrowsaddingtoone=0;
    end
    if isfield(vfoptions,'outputkron')==0
        vfoptions.outputkron=0;
    end
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if isfield(vfoptions,'V0')
    V0=reshape(vfoptions.V0,[N_a,N_z]);
else
    if vfoptions.parallel==2
        V0=zeros([N_a,N_z], 'gpuArray');
    else
        V0=zeros([N_a,N_z]);
    end
end

%% Check the sizes of some of the inputs
if strcmp(vfoptions.solnmethod,'purediscretization') || strcmp(vfoptions.solnmethod,'localpolicysearch')
    if size(d_grid)~=[sum(n_d), 1]
        error('d_grid is not the correct shape (should be of size sum(n_d)-by-1)')
    elseif size(a_grid)~=[sum(n_a), 1]
        error('a_grid is not the correct shape (should be of size sum(n_a)-by-1)')
    elseif N_z>0
        if size(z_grid)~=[sum(n_z), 1]
            if isequal(size(z_grid),[prod(n_z),length(n_z)])
                % Using joint grids
            else
                error('z_grid is not the correct shape (should be of size sum(n_z)-by-1)')
            end
        elseif size(pi_z)~=[N_z, N_z]
            error('pi is not of size N_z-by-N_z')
        end
    elseif n_z(end)>1 % Ignores this final check if last dimension of n_z is singleton as will cause an error
        if ndims(V0)>2
            if size(V0)~=[n_a,n_z] % Allow for input to be already transformed into Kronecker form
                error('Starting choice for ValueFn is not of size [n_a,n_z]')
            end
        elseif size(V0)~=[N_a,N_z] % Allows for possiblity that V0 is already in kronecker form
            error('Starting choice for ValueFn is not of size [n_a,n_z]')
        end
    end
end

if N_z>0
    if min(min(pi_z))<0
        error('Problem with pi_z in ValueFnIter_Case1: min(min(pi_z))<0 \n')
    elseif vfoptions.piz_strictonrowsaddingtoone==1
        if max(sum(pi_z,2))~=1 || min(sum(pi_z,2))~=1
            error('Problem with pi_z in ValueFnIter_Case1: rows do not sum to one \n')
        end
    elseif vfoptions.piz_strictonrowsaddingtoone==0
        if max(abs((sum(pi_z,2))-1)) > 10^(-13)
            error('Problem with pi_z in ValueFnIter_Case1: rows do not sum to one \n')
        end
    end
end

if max(vfoptions.endotype)==1
    if ~strcmp(vfoptions.solnmethod,'purediscretization_refinement2') 
        error('Using vfoptions.endotype only works with vfoptions.solnmethod as purediscretization_refinement2')
    end
end
if max(vfoptions.incrementaltype)==1
    if ~strcmp(vfoptions.solnmethod,'purediscretization') 
        error('Using vfoptions.incrementaltype only works with vfoptions.solnmethod as purediscretization')
    end
end



%% Implement new way of handling ReturnFn inputs
if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_a_temp=l_a;
l_z=length(n_z);
l_z_temp=l_z;
if max(vfoptions.endotype)==1
    l_a_temp=l_a-sum(vfoptions.endotype); % Some of the endogenous states is an endogenous type, so it won't appear at this 
    l_z_temp=l_z+sum(vfoptions.endotype); % The variables after z is the endogenous types
end
% If no ReturnFnParamNames inputted, then figure it out from ReturnFn
if isempty(ReturnFnParamNames)
    temp=getAnonymousFnInputNames(ReturnFn);
    if length(temp)>(l_d+l_a_temp+l_a_temp+l_z_temp)
        ReturnFnParamNames={temp{l_d+l_a_temp+l_a_temp+l_z_temp+1:end}}; % the first inputs will always be (d,aprime,a,z)
    else
        ReturnFnParamNames={};
    end
% else
%     ReturnFnParamNames=ReturnFnParamNames;
end

%%

if vfoptions.parallel==2 
   % If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
   V0=gpuArray(V0);
   pi_z=gpuArray(pi_z);
   d_grid=gpuArray(d_grid);
   a_grid=gpuArray(a_grid);
   z_grid=gpuArray(z_grid);
% else
%    % If using CPU make sure all the relevant inputs are CPU arrays (not standard arrays)
%    % This may be completely unnecessary.
%    V0=gather(V0);
%    pi_z=gather(pi_z);
%    d_grid=gather(d_grid);
%    a_grid=gather(a_grid);
%    z_grid=gather(z_grid);
end

if vfoptions.verbose==1
    vfoptions
end

%% Alternative solution methods
if strcmp(vfoptions.solnmethod,'localpolicysearch') 
    % Solve value function using 'local policy search' method.
    [V, Policy]=ValueFnIter_Case1_LPS(V0, n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z, DiscountFactorParamNames, ReturnFn, Parameters, ReturnFnParamNames, vfoptions);
    varargout={V,Policy};
    return
end

% % fVFI using Chebyshev policynomials and smolyak grids
% if strcmp(vfoptions.solnmethod,'smolyak_chebyshev') 
%     % Solve value function using smolyak grids and chebyshev polynomials (see Judd, Maliar, Maliar & Valero (2014).
%     [V, Policy]=ValueFnIter_Case1_SmolyakChebyshev(V0, n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z, DiscountFactorParamNames, ReturnFn, Parameters, ReturnFnParamNames, vfoptions);
%     varargout={V,Policy};
%     return
% end

%% Entry and Exit
if vfoptions.endogenousexit==1
    % ExitPolicy is binary decision to exit (1 is exit, 0 is 'not exit').
    [V, Policy,ExitPolicy]=ValueFnIter_Case1_EndogExit(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    varargout={V,Policy,ExitPolicy};
    return
elseif vfoptions.endogenousexit==2 % Mixture of endogenous and exogenous exit.
    % ExitPolicy is binary decision to exit (1 is exit, 0 is 'not exit').
    % Policy is for those who remain.
    % PolicyWhenExit is current period decisions of those who will exit at end of period.
    [V, Policy, PolicyWhenExit, ExitPolicy]=ValueFnIter_Case1_EndogExit2(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    varargout={V,Policy, PolicyWhenExit,ExitPolicy};
    return
end

%% Exotic Preferences
if isfield(vfoptions,'exoticpreferences')
    if strcmp(vfoptions.exoticpreferences,'None')
        % Just ignore and will then continue on.
    elseif strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
        [V, Policy]=ValueFnIter_Case1_QuasiHyperbolic(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, vfoptions,Parameters,ReturnFnParamNames);
        varargout={V,Policy};
        return
    elseif strcmp(vfoptions.exoticpreferences,'EpsteinZin')
        [V, Policy]=ValueFnIter_Case1_EpsteinZin(V0, n_d,n_a,n_z,d_grid,a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, vfoptions,Parameters,ReturnFnParamNames);
        varargout={V,Policy};
        return
    elseif vfoptions.exoticpreferences==3 % Allow the discount factor to depend on the (next period) exogenous state.
        % To implement this, can actually just replace the discount factor by 1, and adjust pi_z appropriately.
        % Note that distinguishing the discount rate and pi_z is important in almost all other contexts. Just not in this one.
        
        % Create a matrix containing the DiscountFactorParams,
        nDiscFactors=length(DiscountFactorParamNames);
        DiscountFactorParamsMatrix=Parameters.(DiscountFactorParamNames{1});
        if nDiscFactors>1
            for ii=2:nDiscFactors
                DiscountFactorParamsMatrix=DiscountFactorParamsMatrix.*(Parameters.(DiscountFactorParamNames{ii}));
            end
        end
        DiscountFactorParamsMatrix=DiscountFactorParamsMatrix.*ones(N_z,N_z); % Make it of size z-by-zprime, so that I can later just assume that it takes this shape
        if vfoptions.parallel==2
            DiscountFactorParamsMatrix=gpuArray(DiscountFactorParamsMatrix);
        end
        % Set the 'fake discount factor to one.
        DiscountFactorParamsVec=1;
        % Set pi_z to include the state-dependent discount factors
        pi_z=pi_z.*DiscountFactorParamsMatrix;
    end
end

%% State Dependent Parameters
n_SDP=0;
SDP1=[]; SDP2=[]; SDP3=[];
if isfield(vfoptions,'statedependentparams')
    % Remove the statedependentparams from ReturnFnParamNames
    ReturnFnParamNames=setdiff(ReturnFnParamNames,vfoptions.statedependentparams.names);
    % Note that the codes assume that the statedependentparams are the first elements in ReturnFnParamNames
    % Codes currently allow up to three state dependent parameters
    n_SDP=length(vfoptions.statedependentparams.names);
    if N_d>1
        l_d=length(n_d);
        n_full=[n_d,n_a,n_a,n_z];
    else
        l_d=0;
        n_full=[n_a,n_a,n_z];
    end
    l_a=length(n_a);
    l_z=length(n_z);
    
    % First state dependent parameter, get into form needed for the valuefn
    SDP1=Params.(vfoptions.statedependentparams.names{1});
    SDP1_dims=vfoptions.statedependentparams.dimensions.(vfoptions.statedependentparams.names{1});
%     vfoptions.statedependentparams.dimensions.kmax=[3,4,5,6,7]; % The d,a & z variables (in VFI toolkit notation)
    temp=ones(1,l_d+l_a+l_a+l_z);
    for jj=1:max(SDP1_dims)
        [v,ind]=max(SDP1_dims==jj);
        if v==1
            temp(jj)=n_full(ind);
        end
    end
    if isscalar(SDP1)
        SDP1=SDP1*ones(temp);
    else
        SDP1=reshape(SDP1,temp);
    end
    if n_SDP>=2
        % Second state dependent parameter, get into form needed for the valuefn
        SDP2=Params.(vfoptions.statedependentparams.names{2});
        SDP2_dims=vfoptions.statedependentparams.dimensions.(vfoptions.statedependentparams.names{2});
        temp=ones(1,l_d+l_a+l_a+l_z);
        for jj=1:max(SDP2_dims)
            [v,ind]=max(SDP2_dims==jj);
            if v==1
                temp(jj)=n_full(ind);
            end
        end
        if isscalar(SDP2)
            SDP2=SDP2*ones(temp);
        else
            SDP2=reshape(SDP2,temp);
        end
    end
    if n_SDP>=3
        % Third state dependent parameter, get into form needed for the valuefn
        SDP3=Params.(vfoptions.statedependentparams.names{3});
        SDP3_dims=vfoptions.statedependentparams.dimensions.(vfoptions.statedependentparams.names{3});
        temp=ones(1,l_d+l_a+l_a+l_z);
        for jj=1:max(SDP3_dims)
            [v,ind]=max(SDP3_dims==jj);
            if v==1
                temp(jj)=n_full(ind);
            end
        end
        if isscalar(SDP3)
            SDP3=SDP3*ones(temp);
        else
            SDP3=reshape(SDP3,temp);
        end
    end
    if n_SDP>3
        fprintf('WARNING: currently only three state dependent parameters are allowed. If you have a need for more please email robertdkirkby@gmail.com and let me know (I can easily implement more if needed) \n')
        dbstack
        return
    end
end

%% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
if isfield(vfoptions,'exoticpreferences')
    if vfoptions.exoticpreferences~=3
        DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
        if vfoptions.exoticpreferences==0
            DiscountFactorParamsVec=prod(DiscountFactorParamsVec); % Infinite horizon, so just do this once.
        end
    end
else
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec); % Infinite horizon, so just do this once.
end

%%
if strcmp(vfoptions.solnmethod,'purediscretization_relativeVFI') 
    % Note: have only implemented Relative VFI on the GPU
    warning('Relative VFI is unstable if you have substantial discretization (has difficulty converging if you dont use enough points)')
    [VKron,Policy]=ValueFnIter_Case1_RelativeVFI(V0,n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z,ReturnFn,ReturnFnParamsVec,DiscountFactorParamsVec,vfoptions,n_SDP,SDP1,SDP2,SDP3);
end

%%
if strcmp(vfoptions.solnmethod,'purediscretization_endogenousVFI') 
    % Note: have only implemented Endogenous VFI on the GPU
    error('Endogenous VFI is not yet working')
%     [VKron,Policy]=ValueFnIter_Case1_EndoVFI(V0,n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z,ReturnFn,ReturnFnParamsVec,DiscountFactorParamsVec,vfoptions,n_SDP,SDP1,SDP2,SDP3);
end

%% Semi-endogenous state
% The transition matrix of the exogenous shocks depends on the value of the endogenous state.
if isfield(vfoptions,'SemiEndogShockFn')
    if vfoptions.lowmemory~=0 || vfoptions.parallel<1 % GPU or parellel CPU are only things that I have created (email me if you want/need other options)
        error('Only lowmemory=0 and parallel=1 or 2 are currently possible when using vfoptions.SemiEndogShockFn \n')
    end
    if vfoptions.verbose==1
        fprintf('Creating return fn matrix \n')
    end
    if vfoptions.returnmatrix==0
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, vfoptions.parallel, ReturnFnParamsVec);
    elseif vfoptions.returnmatrix==1
        ReturnMatrix=ReturnFn;
    elseif vfoptions.returnmatrix==2 % GPU
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
    end
    if vfoptions.verbose==1
        time=toc;
        fprintf('Time to create return fn matrix: %8.4f \n', time)
        fprintf('Starting pi_z_endog \n')
        tic;
    end
    
    if isa(vfoptions.SemiEndogShockFn,'function_handle')==0
        pi_z_semiendog=vfoptions.SemiEndogShockFn;
    else
        if ~isfield(vfoptions,'SemiEndogShockFnParamNames')
            error('vfoptions.SemiEndogShockFnParamNames is missing (is needed for vfoptions.SemiEndogShockFn) \n')
        end
        pi_z_semiendog=zeros(N_a,N_z,N_z);
        a_gridvals=CreateGridvals(n_a,a_grid,2);
        SemiEndogParamsVec=CreateVectorFromParams(Parameters, vfoptions.SemiEndogShockFnParamNames);
        SemiEndogParamsCell=cell(length(SemiEndogParamsVec),1);
        for ii=1:length(SemiEndogParamsVec)
            SemiEndogParamsCell(ii,1)={SemiEndogParamsVec(ii)};
        end
        parfor ii=1:N_a
            a_ii=a_gridvals(ii,:)';
            a_ii_SemiEndogParamsCell=[a_ii;SemiEndogParamsCell];
            [~,temp_pi_z]=SemiEndogShockFn(a_ii_SemiEndogParamsCell{:});
            pi_z_semiendog(ii,:,:)=temp_pi_z;
            % Note that temp_z_grid is just the same things for all k, and same as
            % z_grid created about 10 lines above, so I don't bother keeping it.
            % I only create it so you can double-check it is same as z_grid
        end
    end
    if vfoptions.verbose==1
        time=toc;
        fprintf('Time to create semi-endogenous shock transition matrix: %8.4f \n', time)
        fprintf('Starting Value Function \n')
        tic;
    end
    
    if vfoptions.parallel==2
        if n_d(1)==0
            [VKron,Policy]=ValueFnIter_Case1_NoD_SemiEndog_Par2_raw(V0, n_a, n_z, pi_z_semiendog, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
        else
            [VKron, Policy]=ValueFnIter_Case1_SemiEndog_Par2_raw(V0, n_d, n_a, n_z, pi_z_semiendog, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
        end
    elseif vfoptions.parallel==1
        if n_d(1)==0
            [VKron,Policy]=ValueFnIter_Case1_NoD_SemiEndog_Par1_raw(V0, n_a, n_z, pi_z_semiendog, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
        else
            [VKron, Policy]=ValueFnIter_Case1_SemiEndog_Par1_raw(V0, n_d, n_a, n_z, pi_z_semiendog, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
        end        
    end
    if vfoptions.outputkron==0
        V=reshape(VKron,[n_a,n_z]);
        Policy=UnKronPolicyIndexes_Case1(Policy, n_d, n_a, n_z,vfoptions);
        if vfoptions.verbose==1
            time=toc;
            fprintf('Time to create UnKron Value Fn and Policy: %8.4f \n', time)
        end
    else
        varargout={VKron,Policy};
        return
    end
    
    if vfoptions.polindorval==2
        Policy=PolicyInd2Val_Case1(Policy,n_d,n_a,n_z,d_grid, a_grid,vfoptions.parallel);
    end
    
    % Sometimes numerical rounding errors (of the order of 10^(-16) can mean
    % that Policy is not integer valued. The following corrects this by converting to int64 and then
    % makes the output back into double as Matlab otherwise cannot use it in
    % any arithmetical expressions.
    if vfoptions.policy_forceintegertype==1
        Policy=uint64(Policy);
        Policy=double(Policy);
    end
    
    varargout={V,Policy};
    return
end

%% Detect if using incremental endogenous states and solve this using purediscretization, prior to the main purediscretization routines
if max(vfoptions.incrementaltype==1) && strcmp(vfoptions.solnmethod,'purediscretization')
    % Incremental Endogenous States: aprime either equals a, or one grid point higher (unchanged on incremental increase)
    [VKron,Policy]=ValueFnIter_Case1_Increment(V0,n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z,ReturnFn,ReturnFnParamsVec,DiscountFactorParamsVec,vfoptions);
end

%%
if strcmp(vfoptions.solnmethod,'purediscretization') 
    if vfoptions.parallel==1 && vfoptions.lowmemory==2
        fprintf('Use of vfoptions.lowmemory=2 in not supported for cpu, have switched to vfoptions.lowmemory=1 \n')
        vfoptions.lowmemory=1;
    end

    if vfoptions.lowmemory==0
        
        %% CreateReturnFnMatrix_Case1_Disc creates a matrix of dimension (d and aprime)-by-a-by-z.
        % Since the return function is independent of time creating it once and
        % then using it every iteration is good for speed, but it does use a lot of memory.
        
        if vfoptions.verbose==1
            disp('Creating return fn matrix')
            tic;
            if vfoptions.returnmatrix==0
                fprintf('NOTE: When using CPU you can speed things up by giving return fn as a matrix; see vfoptions.returnmatrix=1 in VFI Toolkit documentation. \n')
            end
        end
        
        if isfield(vfoptions,'statedependentparams')
            if vfoptions.returnmatrix==2 % GPU
                if n_SDP==3
                    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2_SDP(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec,SDP1,SDP2,SDP3);
                elseif n_SDP==2
                    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2_SDP(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec,SDP1,SDP2);
                elseif n_SDP==1
                    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2_SDP(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec,SDP1);
                end
            else
                fprintf('ERROR: statedependentparams only works with GPU (parallel=2) \n')
                dbstack
            end
        else % Following is the normal/standard behavior
            if vfoptions.returnmatrix==0
                ReturnMatrix=CreateReturnFnMatrix_Case1_Disc(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, vfoptions.parallel, ReturnFnParamsVec);
            elseif vfoptions.returnmatrix==1
                ReturnMatrix=ReturnFn;
            elseif vfoptions.returnmatrix==2 % GPU
                ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
            end
        end
        
        if vfoptions.verbose==1
            time=toc;
            fprintf('Time to create return fn matrix: %8.4f \n', time)
            fprintf('Starting Value Function \n')
            tic;
        end
        
        
        if n_d(1)==0
            if vfoptions.parallel==0     % On CPU
                [VKron,Policy]=ValueFnIter_Case1_NoD_raw(V0, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
            elseif vfoptions.parallel==1 % On Parallel CPU
                [VKron,Policy]=ValueFnIter_Case1_NoD_Par1_raw(V0, N_a, N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
            elseif vfoptions.parallel==2 % On GPU
%                 [VKron,Policy]=ValueFnIter_Case1_NoD_Par2_Alt_raw(V0, n_a, n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance); %  a_grid, z_grid,
                [VKron,Policy]=ValueFnIter_Case1_NoD_Par2_raw(V0, n_a, n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance); %  a_grid, z_grid,
            end
        else
            if vfoptions.parallel==0 % On CPU
                [VKron, Policy]=ValueFnIter_Case1_raw(V0, N_d,N_a,N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
            elseif vfoptions.parallel==1 % On Parallel CPU
                [VKron, Policy]=ValueFnIter_Case1_Par1_raw(V0, N_d,N_a,N_z, pi_z, DiscountFactorParamsVec, ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
            elseif vfoptions.parallel==2 % On GPU
                [VKron, Policy]=ValueFnIter_Case1_Par2_raw(V0, n_d,n_a,n_z, pi_z, DiscountFactorParamsVec, ReturnMatrix,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
            end
        end
        
    elseif vfoptions.lowmemory==1
                
        if vfoptions.verbose==1
            disp('Starting Value Function')
            tic;
        end
        
        if n_d(1)==0
            if vfoptions.parallel==0
                [VKron,Policy]=ValueFnIter_Case1_LowMem_NoD_raw(V0, n_a, n_z, a_grid, z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
            elseif vfoptions.parallel==1
                [VKron,Policy]=ValueFnIter_Case1_LowMem_NoD_Par1_raw(V0, n_a, n_z, a_grid, z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.verbose);
            elseif vfoptions.parallel==2 % On GPU
                [VKron,Policy]=ValueFnIter_Case1_LowMem_NoD_Par2_raw(V0, n_a, n_z, a_grid, z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance);
            end
        else
            if vfoptions.parallel==0
                [VKron, Policy]=ValueFnIter_Case1_LowMem_raw(V0, n_d,n_a,n_z, d_grid,a_grid,z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
            elseif vfoptions.parallel==1
                [VKron, Policy]=ValueFnIter_Case1_LowMem_Par1_raw(V0, n_d,n_a,n_z, d_grid,a_grid,z_grid,pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, vfoptions.verbose);
            elseif vfoptions.parallel==2 % On GPU
                [VKron, Policy]=ValueFnIter_Case1_LowMem_Par2_raw(V0, n_d,n_a,n_z, d_grid, a_grid, z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec,vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance);
            end
        end
        
    elseif vfoptions.lowmemory==2
                
        if vfoptions.verbose==1
            disp('Starting Value Function')
            tic;
        end
        
        if n_d(1)==0
            if vfoptions.parallel==2
                [VKron,Policy]=ValueFnIter_Case1_LowMem2_NoD_Par2_raw(V0, n_a, n_z, a_grid, z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions.howards, vfoptions.tolerance, vfoptions.verbose);
            end
        else
            if vfoptions.parallel==2 % On GPU
                [VKron, Policy]=ValueFnIter_Case1_LowMem2_Par2_raw(V0, n_d,n_a,n_z, d_grid, a_grid, z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec ,vfoptions.howards,vfoptions.tolerance,vfoptions.verbose);
            end
        end
        
    elseif vfoptions.lowmemory==3 % Specifically for the Hayek method where we include prices
        V0=reshape(V0,[N_a,N_z]);
        
        if vfoptions.verbose==1
            disp('Starting Value Function')
            tic;
        end
        
        if n_d(1)==0
            if vfoptions.parallel==2
                [VKron,Policy]=ValueFnIter_Case1_LowMem3_NoD_Par2_raw(V0, n_a, n_z, a_grid, z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions.howards, vfoptions.maxhowards, vfoptions.tolerance, vfoptions.lowmemorydimensions);
            end
        else
            if vfoptions.parallel==2 % On GPU
                [VKron, Policy]=ValueFnIter_Case1_LowMem3_Par2_raw(V0,   n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec , vfoptions.howards, vfoptions.maxhowards,vfoptions.tolerance, vfoptions.lowmemorydimensions);
            end
        end
    end
end

%%
if strcmp(vfoptions.solnmethod,'purediscretization_refinement') 
    if n_d(1)==0
        warning('You are using purediscretization_refinement as the vfoptions.solnmethod, but you have no decision (d) variables, this is likely innappropriate (will work but unnecessary overhead/slower)')
    end
    % Refinement: Presolve for dstar(aprime,a,z). Then solve value function for just aprime,a,z. 
    [VKron,Policy]=ValueFnIter_Case1_Refine(V0,n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z,ReturnFn,ReturnFnParamsVec,DiscountFactorParamsVec,vfoptions);
end

if strcmp(vfoptions.solnmethod,'purediscretization_refinement2') 
    if n_d(1)==0
        warning('You are using purediscretization_refinement2 as the vfoptions.solnmethod, but you have no decision (d) variables, this is likely innappropriate (will work but unnecessary overhead/slower)')
    end
    % Refinement: Presolve for dstar(aprime,a,z). Then solve value function for just aprime,a,z. 
    % Refinement 2: Multigrid approach when presolving for dstar(aprime,a,z).
    
    % Check that the info about layers is provided
    if ~isfield(vfoptions,'refine_pts') % points per dimension per layer
        error('Using vfoptions.solnmethod purediscretization_refinement2 you must declare vfoptions.refine_pts')
    else
        if rem(vfoptions.refine_pts,2)~=1
            error('vfoptions.refine_pts must be an odd number')
        end
    end
    if ~isfield(vfoptions,'refine_iter') % number of layers
        error('Using vfoptions.solnmethod purediscretization_refinement2 you must declare vfoptions.refine_iter')
    end
    
    % Check that grid size for d variables matches the ptsperlayer and 
    RequiredGridPoints=nGridPointsWithLayers(vfoptions);
    for ii=1:length(n_d)
        if n_d(ii)~=RequiredGridPoints
            fprintf('Problem with the %i-th decision variable \n',ii)
            fprintf('With current settings for layers (in vfoptions) you should be using %i points for each decision variable \n',RequiredGridPoints)
            error('The number of points in the grid for the i-th variable is does not fit layers')
        end
    end
    
    if max(vfoptions.endotype)==0 % If they are all zeros, no endo types are used
        [VKron,Policy]=ValueFnIter_Case1_Refine2(V0,l_d,N_a,N_z,n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z,ReturnFn,ReturnFnParamsVec,DiscountFactorParamsVec,vfoptions);
    else
        % Need to seperate endogenous states from endogenous types to take advantage of them
        n_endostate=n_a(logical(1-vfoptions.endotype));
        n_endotype=n_a(logical(vfoptions.endotype));
        endostate_grid=zeros(sum(n_endostate),1);
        endotype_grid=zeros(sum(n_endotype),1);
        endostate_c=1;
        endotype_c=1;
        if vfoptions.endotype(1)==1 % Endogenous type
            endotype_grid(1:n_a(1))=a_grid(1:n_a(1));
            endotype_c=endotype_c+1;
        else % Endogenous state
            endostate_grid(1:n_a(1))=a_grid(1:n_a(1));
            endostate_c=endostate_c+1;
        end
        for ii=2:length(n_a)
            if vfoptions.endotype(ii)==1 % Endogenous type
                if endotype_c==1
                    endotype_grid(1:n_endotype(1))=a_grid(1+sum(n_a(1:ii-1)):sum(n_a(1:ii)));
                else
                    endotype_grid(1+sum(n_endotype(1:endotype_c-1)):sum(n_endotype(1:endotype_c)))=a_grid(1+sum(n_a(1:ii-1)):sum(n_a(1:ii)));
                end
                endotype_c=endotype_c+1;
            else % Endogenous state
                if endotype_c==1
                    endostate_grid(1:n_endostate(1))=a_grid(1+sum(n_a(1:ii-1)):sum(n_a(1:ii)));
                else
                    endostate_grid(1+sum(n_endostate(1:endostate_c-1)):sum(n_endostate(1:endostate_c)))=a_grid(1+sum(n_a(1:ii-1)):sum(n_a(1:ii)));
                end
                endostate_c=endostate_c+1;
            end
        end
        
        [VKron,Policy]=ValueFnIter_Case1_EndoType_Refine2(V0,l_d,prod(n_endostate),N_z,n_d,n_endostate,n_z,n_endotype,d_grid,endostate_grid,z_grid,endotype_grid,pi_z,ReturnFn,ReturnFnParamsVec,DiscountFactorParamsVec,vfoptions);
    end
    % To be able to resize the output we need to treat endotype is just
    % another endogenous state. This will happen because of how we have n_a setup.
end

%%
if strcmp(vfoptions.solnmethod,'purediscretization_PFI') 
    % Note: have only implemented PFI on the GPU
    [VKron,Policy]=ValueFnIter_Case1_PolicyFnIter(V0,n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z,ReturnFn,ReturnFnParamsVec,DiscountFactorParamsVec,vfoptions,n_SDP,SDP1,SDP2,SDP3);
end

if vfoptions.verbose==1
    time=toc;
    fprintf('Time to solve for Value Fn and Policy: %8.4f \n', time)
    disp('Transforming Value Fn and Optimal Policy matrices back out of Kronecker Form')
    tic;
end
%% Cleaning up the output
if vfoptions.outputkron==0
    V=reshape(VKron,[n_a,n_z]);
    Policy=UnKronPolicyIndexes_Case1(Policy, n_d, n_a, n_z,vfoptions);
    if vfoptions.verbose==1
        time=toc;
        fprintf('Time to create UnKron Value Fn and Policy: %8.4f \n', time)
    end
else
    varargout={VKron,Policy};
    return
end

if vfoptions.polindorval==2
    Policy=PolicyInd2Val_Case1(Policy,n_d,n_a,n_z,d_grid, a_grid);
end
    
% Sometimes numerical rounding errors (of the order of 10^(-16) can mean
% that Policy is not integer valued. The following corrects this by converting to int64 and then
% makes the output back into double as Matlab otherwise cannot use it in
% any arithmetical expressions.
if vfoptions.policy_forceintegertype==1
    Policy=uint64(Policy);
    Policy=double(Policy);
end

varargout={V,Policy};

end