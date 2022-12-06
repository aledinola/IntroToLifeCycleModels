function SimPanelValues=SimPanelValues_Case1(InitialDist,Policy,FnsToEvaluate,FnsToEvaluateParamNames,Parameters,n_d,n_a,n_z,d_grid,a_grid,z_grid,pi_z, simoptions, EntryExitParamNames,PolicyWhenExiting)
% Simulates a panel based on PolicyIndexes of 'numbersims' agents of length
% 'simperiods' beginning from randomly drawn InitialDist.
% SimPanelValues is a 2-dimensional matrix with first dimension being the
% number of 'variables' to be simulated, second dimension is J, and
% third dimension is the number-of-simulations.
%
% Note that when there is entry of new agents the number-of-simulations
% will be larger than simoptions.numbersims.
%
% InitialDist can be inputed as over the finite time-horizon (j), or
% without a time-horizon in which case it is assumed to be an InitialDist
% for time j=1. (So InitialDist is either n_a-by-n_z-by-n_j, or n_a-by-n_z)

N_a=prod(n_a);
N_z=prod(n_z);
N_d=prod(n_d);

%% Check which simoptions have been declared, set all others to defaults 
if exist('simoptions','var')==1
    %Check simoptions for missing fields, if there are some fill them with the defaults
    if isfield(simoptions,'parallel')==0
        simoptions.parallel=1+(gpuDeviceCount>0);
    end
    if isfield(simoptions,'verbose')==0
        simoptions.verbose=0;
    end
    if isfield(simoptions,'simperiods')==0
        simoptions.simperiods=50;
    end
    if isfield(simoptions,'numbersims')==0
        simoptions.numbersims=10^3;
    end
    if isfield(simoptions,'agententryandexit')==0
        simoptions.agententryandexit=0;
    end
    if isfield(simoptions,'endogenousexit')==0
        simoptions.endogenousexit=0; % Note: this will only be relevant if agententryandexit=1
    end
    if isfield(simoptions,'entryinpanel')==0
        if simoptions.agententryandexit==1
            simoptions.entryinpanel=1; % Note: this will only be relevant if agententryandexit=1
        else
            simoptions.entryinpanel=0;
        end
    end
    if isfield(simoptions,'exitinpanel')==0
        if simoptions.agententryandexit==1
            simoptions.exitinpanel=1; % Note: this will only be relevant if agententryandexit=1
        else
            simoptions.exitinpanel=0;
        end
    end
else
    %If simoptions is not given, just use all the defaults
    simoptions.parallel=1+(gpuDeviceCount>0);
    simoptions.verbose=0;
    simoptions.simperiods=50;
    simoptions.numbersims=10^3;
    simoptions.agententryandexit=0;
    simoptions.endogenousexit=0; % Note: this will only be relevant if agententryandexit=1
    simoptions.entryinpanel=0; % Note: this will only be relevant if agententryandexit=1
    simoptions.exitinpanel=0; % Note: this will only be relevant if agententryandexit=1
end

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

if simoptions.parallel~=2
    d_grid=gather(d_grid);
    a_grid=gather(a_grid);
    z_grid=gather(z_grid);
%     pi_z=gather(pi_z);
%     Policy=gather(Policy);
%     InitialDist.pdf=gather(InitialDist.pdf);
end

if simoptions.agententryandexit==1 && isfield(simoptions,'SemiEndogShockFn')
    fprintf('ERROR: Cannot currently use simoptions.agententryandexit==1 SemiEndogShockFn together. Email me if you need this. \n')
    dbstack
    return
end

%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
        if length(temp)>(l_d+l_a+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(AggVarNames{ff});
    end    
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end


%% Simulate SimPanelIndexes_Case1

% NOTE: ESSENTIALLY ALL THE RUN TIME IS IN THIS COMMAND. WOULD BE GOOD TO OPTIMIZE/IMPROVE.
PolicyIndexesKron=KronPolicyIndexes_Case1(Policy, n_d, n_a, n_z);%,simoptions); % Create it here as want it both here and inside SimPanelIndexes_FHorz_Case1 (which will recognise that it is already in this form)
% if exists(PolicyWhenExiting,'var')
%     PolicyWhenExitingKron=KronPolicyIndexes_Case1(PolicyWhenExiting, n_d, n_a, n_z);
% end

if simoptions.agententryandexit==1
    % Do everything for an extra period, and then delete this at the end
    % (this is needed as lot's of info about exit decisions gets encoded
    % into the next period as a way to minimize memory usage)
    simoptions.simperiods=simoptions.simperiods+1;
    
    DistOfNewAgents=gather(Parameters.(EntryExitParamNames.DistOfNewAgents{1}));
    CondlProbOfSurvival=gather(Parameters.(EntryExitParamNames.CondlProbOfSurvival{1}));
    RelativeMassOfEntrants=Parameters.(EntryExitParamNames.MassOfNewAgents{1})/InitialDist.mass;

    % Rather than create a whole new function for Entry, just deal with it
    % by making repeated use of SimPanelIndexes_Case1(). This could be sped
    % up with better use of precomputing certain objects, but is easy.
    
    % First, figure out how big the eventual panel will be.
    NumberOfNewAgentsPerPeriod=round(RelativeMassOfEntrants*simoptions.numbersims);
    if simoptions.entryinpanel==0 % Don't want entry in panel data simulation
        NumberOfNewAgentsPerPeriod=0;
    end
    TotalNumberSims=simoptions.numbersims+simoptions.simperiods*NumberOfNewAgentsPerPeriod;
    SimPanelIndexes=nan(l_a+l_z,simoptions.simperiods,TotalNumberSims); % (a,z)
    % Start with those based on the initial distribution
    SimPanelIndexes(:,:,1:simoptions.numbersims)=gather(SimPanelIndexes_Case1(InitialDist.pdf,PolicyIndexesKron,n_d,n_a,n_z,pi_z, simoptions, CondlProbOfSurvival, Parameters));
    % Now do those for the entrants each period
    numbersims=simoptions.numbersims; % Store this, so can restore it after following loop
    simperiods=simoptions.simperiods;% Store this, so can restore it after following loop
    simoptions.numbersims=NumberOfNewAgentsPerPeriod;
    for t=1:simperiods
        SimPanelIndexes(:,t:end,numbersims+1+NumberOfNewAgentsPerPeriod*(t-1):numbersims+NumberOfNewAgentsPerPeriod*t)=gather(SimPanelIndexes_Case1(DistOfNewAgents,PolicyIndexesKron,n_d,n_a,n_z,pi_z, simoptions, CondlProbOfSurvival, Parameters));
        simoptions.simperiods=simoptions.simperiods-1;
    end
    simoptions.numbersims=numbersims; % Restore.
    simoptions.simperiods=simperiods;% Retore.
elseif isfield(simoptions,'SemiEndogShockFn')
    SimPanelIndexes=SimPanelIndexes_Case1_SemiEndog(InitialDist,PolicyIndexesKron,n_d,n_a,n_z,pi_z, simoptions);
else % simoptions.agententryandexit==0    
    SimPanelIndexes=SimPanelIndexes_Case1(InitialDist,PolicyIndexesKron,n_d,n_a,n_z,pi_z, simoptions);
end

%% From now on is just replacing the indexes with values
% Move everything to cpu for what remains.
d_grid=gather(d_grid);
a_grid=gather(a_grid);
z_grid=gather(z_grid);
% PolicyIndexesKron=gather(PolicyIndexesKron);

SimPanelValues=zeros(length(FnsToEvaluate), simoptions.simperiods, size(SimPanelIndexes,3)); % Normally size(SimPanelIndexes,3) will equal simoptions.numbersims, but not when there is entry.

%% Precompute the gridvals vectors.
if simoptions.agententryandexit==1 && simoptions.endogenousexit==1
    % Policy contains zeros relating to aprime_ind endogenous exit
    % These zeros would cause problem for CreateGridvals_Policy(), but are
    % not needed for anything here. So can just replace with nan.
    Policy(Policy==0)=nan;
    [d_gridvals, aprime_gridvals]=CreateGridvals_Policy_IgnoringNan(Policy,n_d,n_a,n_a,n_z,d_grid,a_grid,1,2);
else
    [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(Policy,n_d,n_a,n_a,n_z,d_grid,a_grid,1,2);
end
a_gridvals=CreateGridvals(n_a,a_grid,2); % 1 at end indicates output as matrices.
if all(size(z_grid)==[sum(n_z),1])
    z_gridvals=CreateGridvals(n_z,z_grid,1); % 1 at end indicates output as matrices.
elseif all(size(z_grid)==[prod(n_z),lenght(n_z)])
    z_gridvals=z_grid;
end
z_gridvals=num2cell(z_gridvals);

if exist('PolicyWhenExiting','var')
    [d_gridvalsWhenExiting, aprime_gridvalsWhenExiting]=CreateGridvals_Policy(PolicyWhenExiting,n_d,n_a,n_a,n_z,d_grid,a_grid,1,2);
end

% d_val=zeros(1,l_d);
% aprime_val=zeros(1,l_a);
% a_val=zeros(1,l_a);
% z_val=zeros(1,l_z);

%% For sure the following could be made faster by parallelizing some stuff.
if simoptions.agententryandexit==0
    parfor ii=1:simoptions.numbersims
        SimPanel_ii=SimPanelIndexes(:,:,ii);
        SimPanelValues_ii=nan(length(FnsToEvaluate),simoptions.simperiods);
        for t=1:simoptions.simperiods
            a_sub=SimPanel_ii(1:l_a,t);
            a_ind=sub2ind_homemade(n_a,a_sub);
            
            z_sub=SimPanel_ii((l_a+1):(l_a+l_z),t);
            z_ind=sub2ind_homemade(n_z,z_sub);
            
            j_ind=SimPanel_ii(end,t); % DO I ACTUALLY WANT THIS HERE? (PRETTY SURE THAT IT IS JUST ACTING AS AN EXTRA INPUT LATER WHICH IS BEING IGNORED AS NOT RELEVANT)
            
            if l_d==0
                for vv=1:length(FnsToEvaluate)
                    if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'FnsToEvaluateParamNames={}'
                        SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(aprime_gridvals{a_ind+(z_ind-1)*N_a,:},a_gridvals{a_ind,:},z_gridvals{z_ind,:});
                    else
                        ValuesFnParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,j_ind));
                        SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(aprime_gridvals{a_ind+(z_ind-1)*N_a,:},a_gridvals{a_ind,:},z_gridvals{z_ind,:},ValuesFnParamsCell{:});
                    end
                end
            else
                for vv=1:length(FnsToEvaluate)
                    if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'FnsToEvaluateParamNames={}'
                        SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(d_gridvals{a_ind+(z_ind-1)*N_a,:},aprime_gridvals{a_ind+(z_ind-1)*N_a,:},a_gridvals{a_ind,:},z_gridvals{z_ind,:});
                    else
                        ValuesFnParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,j_ind));
                        SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(d_gridvals{a_ind+(z_ind-1)*N_a,:},aprime_gridvals{a_ind+(z_ind-1)*N_a,:},a_gridvals{a_ind,:},z_gridvals{z_ind,:},ValuesFnParamsCell{:});
                    end
                end
            end
        end
        SimPanelValues(:,:,ii)=SimPanelValues_ii;
    end
elseif simoptions.agententryandexit==1 && simoptions.endogenousexit==0
    % Need to add check for nan relating to a_ind and z_ind around entry/exit
    for ii=1:size(SimPanelIndexes,3) % simoptions.numbersims
        SimPanelValues_ii=nan(length(FnsToEvaluate),simoptions.simperiods); % Want nan when agents 'die/exit' before end of panel
        SimPanel_ii=SimPanelIndexes(:,:,ii);
        for t=1:simoptions.simperiods
            a_sub=SimPanel_ii(1:l_a,t);
            a_ind=sub2ind_homemade(n_a,a_sub);
            if ~isnan(a_ind)
                
                z_sub=SimPanel_ii((l_a+1):(l_a+l_z),t);
                z_ind=sub2ind_homemade(n_z,z_sub);
                
                j_ind=SimPanel_ii(end,t);
                
                if l_d==0
                    for vv=1:length(FnsToEvaluate)
                        if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'FnsToEvaluateParamNames={}'
                            SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(aprime_gridvals{a_ind+(z_ind-1)*N_a,:},a_gridvals{a_ind,:},z_gridvals{z_ind,:});
                        else
                            ValuesFnParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,j_ind));
                            SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(aprime_gridvals{a_ind+(z_ind-1)*N_a,:},a_gridvals{a_ind,:},z_gridvals{z_ind,:},ValuesFnParamsCell{:});
                        end
                    end
                else
                    for vv=1:length(FnsToEvaluate)
                        if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'FnsToEvaluateParamNames={}'
                            SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(d_gridvals{a_ind+(z_ind-1)*N_a,:},aprime_gridvals{a_ind+(z_ind-1)*N_a,:},a_gridvals{a_ind,:},z_gridvals{z_ind,:});
                        else
                            ValuesFnParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,j_ind));
                            SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(d_gridvals{a_ind+(z_ind-1)*N_a,:},aprime_gridvals{a_ind+(z_ind-1)*N_a,:},a_gridvals{a_ind,:},z_gridvals{z_ind,:},ValuesFnParamsCell{:});
                        end
                    end
                end
            end
        end
        SimPanelValues(:,:,ii)=SimPanelValues_ii;
    end
elseif simoptions.agententryandexit==1 && simoptions.endogenousexit==1
    % Need to add check for nan relating to a_ind and z_ind around entry/exit
    % Need to add check for zeros relating to aprime_ind endogenous exit
    % (don't actually need to do so as these will be nan, have been changed
    % earlier in the current script as is not important here)
    for ii=1:size(SimPanelIndexes,3) % simoptions.numbersims
        SimPanelValues_ii=nan(length(FnsToEvaluate),simoptions.simperiods); % Want nan when agents 'die/exit' before end of panel
        SimPanel_ii=SimPanelIndexes(:,:,ii);
        for t=1:simoptions.simperiods
            a_sub=SimPanel_ii(1:l_a,t);
            a_ind=sub2ind_homemade(n_a,a_sub);
            if ~isnan(a_ind)
                
                z_sub=SimPanel_ii((l_a+1):(l_a+l_z),t);
                z_ind=sub2ind_homemade(n_z,z_sub);
                
                j_ind=SimPanel_ii(end,t);
                
                if l_d==0
                    for vv=1:length(FnsToEvaluate)
                        if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'FnsToEvaluateParamNames={}'
                            SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(aprime_gridvals{a_ind+(z_ind-1)*N_a,:},a_gridvals{a_ind,:},z_gridvals{z_ind,:});
                        else
                            ValuesFnParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,j_ind));
                            SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(aprime_gridvals{a_ind+(z_ind-1)*N_a,:},a_gridvals{a_ind,:},z_gridvals{z_ind,:},ValuesFnParamsCell{:});
                        end
                    end
                else
                    for vv=1:length(FnsToEvaluate)
                        if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'FnsToEvaluateParamNames={}'
                            SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(d_gridvals{a_ind+(z_ind-1)*N_a,:},aprime_gridvals{a_ind+(z_ind-1)*N_a,:},a_gridvals{a_ind,:},z_gridvals{z_ind,:});
                        else
                            ValuesFnParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,j_ind));
                            SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(d_gridvals{a_ind+(z_ind-1)*N_a,:},aprime_gridvals{a_ind+(z_ind-1)*N_a,:},a_gridvals{a_ind,:},z_gridvals{z_ind,:},ValuesFnParamsCell{:});
                        end
                    end
                end
            end
        end
        SimPanelValues(:,:,ii)=SimPanelValues_ii;
    end
elseif simoptions.agententryandexit==1 && simoptions.endogenousexit==2
    % NEED TO FILL THIS PART OUT!!!
    % The kind of exit that occurs at time t is recorded in the time t+1 exogenous state as a value of 0 for endog exit.
    % (Note: so exogenous just leaves nan from then on, endog exit leaves 0 in
    % next period exogenous state and otherwise just leaves nan from then on. Notice that a zero value will throw an error if just treated as a standard index.)

    % Need to add check for nan relating to a_ind and z_ind around entry/exit
    for ii=1:size(SimPanelIndexes,3) % simoptions.numbersims
        SimPanelValues_ii=nan(length(FnsToEvaluate),simoptions.simperiods); % Want nan when agents 'die/exit' before end of panel
        SimPanel_ii=SimPanelIndexes(:,:,ii);
        for t=1:simoptions.simperiods
            a_sub=SimPanel_ii(1:l_a,t);
            a_ind=sub2ind_homemade(n_a,a_sub);
            if ~isnan(a_ind)                
                z_sub=SimPanel_ii((l_a+1):(l_a+l_z),t);
                z_ind=sub2ind_homemade(n_z,z_sub);
                
                j_ind=SimPanel_ii(end,t); 
                
                % Make sure that firm is not currently about to exit (includes where firm faces exogenous exit, even though this is not a decision).
                if t<simoptions.simperiods % Note that with exit the last period will be thrown out anyway, so no need to get it correct.
                    if SimPanel_ii(1:l_a,t+1)~=0 && ~isnan(SimPanel_ii(1:l_a,t+1))
                        exiting=0;
                    else
                        exiting=1;
                    end
                end
                if exiting==0
                    if l_d==0
                        for vv=1:length(FnsToEvaluate)
                            if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'FnsToEvaluateParamNames={}'
                                SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(aprime_gridvals{a_ind+(z_ind-1)*N_a,:},a_gridvals{a_ind,:},z_gridvals{z_ind,:});
                            else
                                ValuesFnParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,j_ind));
                                SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(aprime_gridvals{a_ind+(z_ind-1)*N_a,:},a_gridvals{a_ind,:},z_gridvals{z_ind,:},ValuesFnParamsCell{:});
                            end
                        end
                    else
                        for vv=1:length(FnsToEvaluate)
                            if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'FnsToEvaluateParamNames={}'
                                SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(d_gridvals{a_ind+(z_ind-1)*N_a,:},aprime_gridvals{a_ind+(z_ind-1)*N_a,:},a_gridvals{a_ind,:},z_gridvals{z_ind,:});
                            else
                                ValuesFnParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,j_ind));
                                SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(d_gridvals{a_ind+(z_ind-1)*N_a,:},aprime_gridvals{a_ind+(z_ind-1)*N_a,:},a_gridvals{a_ind,:},z_gridvals{z_ind,:},ValuesFnParamsCell{:});
                            end
                        end
                    end
                elseif exiting==1
                    if l_d==0
                        for vv=1:length(FnsToEvaluate)
                            if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'FnsToEvaluateParamNames={}'
                                SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(aprime_gridvalsWhenExiting{a_ind+(z_ind-1)*N_a,:},a_gridvalsWhenExiting{a_ind,:},z_gridvals{z_ind,:});
                            else
                                ValuesFnParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,j_ind));
                                SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(aprime_gridvalsWhenExiting{a_ind+(z_ind-1)*N_a,:},a_gridvalsWhenExiting{a_ind,:},z_gridvals{z_ind,:},ValuesFnParamsCell{:});
                            end
                        end
                    else
                        for vv=1:length(FnsToEvaluate)
                            if isempty(FnsToEvaluateParamNames(vv).Names)  % check for 'FnsToEvaluateParamNames={}'
                                SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(d_gridvalsWhenExiting{a_ind+(z_ind-1)*N_a,:},aprime_gridvalsWhenExiting{a_ind+(z_ind-1)*N_a,:},a_gridvals{a_ind,:},z_gridvals{z_ind,:});
                            else
                                ValuesFnParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(vv).Names,j_ind));
                                SimPanelValues_ii(vv,t)=FnsToEvaluate{vv}(d_gridvalsWhenExiting{a_ind+(z_ind-1)*N_a,:},aprime_gridvalsWhenExiting{a_ind+(z_ind-1)*N_a,:},a_gridvals{a_ind,:},z_gridvals{z_ind,:},ValuesFnParamsCell{:});
                            end
                        end
                    end
                end
            end
        end
        SimPanelValues(:,:,ii)=SimPanelValues_ii;
    end
end

if simoptions.agententryandexit==1
    % Have done everything for an extra period, and now delete this at the end.
    % (this is needed as lot's of info about exit decisions gets encoded
    % into the next period as a way to minimize memory usage)
    SimPanelValues=SimPanelValues(:,1:end-1,:); % Deletes the extra period
end



%%
if FnsToEvaluateStruct==1
    % Change the output into a structure
    SimPanelValues2=SimPanelValues;
    clear SimPanelValues
    SimPanelValues=struct();
%     AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        SimPanelValues.(AggVarNames{ff})=shiftdim(SimPanelValues2(ff,:,:),1);
    end
end

end



