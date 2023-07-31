function SimLifeCycleProfiles=SimLifeCycleProfiles_FHorz_Case1_PType(InitialDist,PTypeDistParamNames,Policy,FnsToEvaluate,Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid,a_grid,z_grid,pi_z, simoptions)
% Simulates a panel based on PolicyIndexes of 'numbersims' agents of length
% 'simperiods' beginning from randomly drawn InitialDist. Then based on
% this it computes life-cycle profiles for the 'VaulesFns' and reports
% mean, median, min, 19 intermediate ventiles, and max. (you can change from
% ventiles using simperiods.lifecyclepercentiles)
%
% InitialDist can be inputed as over the finite time-horizon (j), or
% without a time-horizon in which case it is assumed to be an InitialDist
% for time/age j=1. (So InitialDist is either n_a-by-n_z-by-n_j-by-n_i, or n_a-by-n_z-by-n_i)

N_a=prod(n_a);
N_z=prod(n_z);
N_d=prod(n_d);


if iscell(Names_i)
    N_i=length(Names_i);
else
    N_i=Names_i; % It is the number of PTypes (which have not been given names)
    Names_i={'ptype001'};
    for ii=2:N_i
        if ii<10
            Names_i{ii}=['ptype00',num2str(ii)];
        elseif ii<100
            Names_i{ii}=['ptype0',num2str(ii)];
        elseif ii<1000
            Names_i{ii}=['ptype',num2str(ii)];
        end
    end
end


%% Check which simoptions have been declared, set all others to defaults 
if exist('simoptions','var')==1
    %Check simoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(simoptions,'parallel')
        simoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
    end
    if ~isfield(simoptions,'simperiods')
        simoptions.simperiods=N_j;
    end
    if ~isfield(simoptions,'numbersims')
        simoptions.numbersims=10^4; % Given that aim is to calculate ventiles of life-cycle profiles 10^4 seems appropriate
    end
    if ~isfield(simoptions,'lifecyclepercentiles')
        simoptions.lifecyclepercentiles=20; % by default gives ventiles
    end
    if isfield(simoptions,'ExogShockFn') % If using ExogShockFn then figure out the parameter names
        simoptions.ExogShockFnParamNames=getAnonymousFnInputNames(simoptions.ExogShockFn);
    end
else
    %If simoptions is not given, just use all the defaults
    simoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    simoptions.verbose=0;
    simoptions.simperiods=N_j;
    simoptions.numbersims=10^4; % Given that aim is to calculate ventiles of life-cycle profiles 10^4 seems appropriate
    simoptions.lifecyclepercentiles=20; % by default gives ventiles
end

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

%%
PTypeWeights=Parameters.(PTypeDistParamNames{1});

for ii=1:N_i
    simoptions_ii=simoptions;
    if simoptions_ii.verbose==1
        sprintf('Fixed type: %i of %i',ii, N_i)
    end

    simoptions_ii.numbersims=round(PTypeWeights(ii)*simoptions.numbersims);
    if isfield(simoptions,'ExogShockFn') % If this exists, so will ExogShockFnParamNames, but I still treat them seperate as makes the code easier to read
        if length(simoptions.ExogShockFn)==1
            if simoptions_ii.ExogShockFn==1
            end
        else
            if simoptions.ExogShockFn(ii)==1
                simoptions_ii.ExogShockFn=simoptions.ExogShockFn(ii);
            end
        end
    end
    if isfield(simoptions,'ExogShockFnParamNames')
        if length(simoptions.ExogShockFnParamNames)==1
            if simoptions.ExogShockFnParamNames==1
            end
        else
            if simoptions.ExogShockFnParamNames(ii)==1
                simoptions_ii.ExogShockFnParamNames=simoptions.ExogShockFnParamNames(ii);
            end
        end
    end
    
    % Go through everything which might be dependent on fixed type (PType)
    % [THIS could be better coded, 'names' are same for all these and just need to be found once outside of ii loop]
    d_grid_temp=d_grid;
    if isa(d_grid,'struct')
        names=fieldnames(d_grid);
        d_grid_temp=d_grid.(names{ii});
    end
    a_grid_temp=a_grid;
    if isa(a_grid,'struct')
        names=fieldnames(a_grid);
        a_grid_temp=a_grid.(names{ii});
    end
    z_grid_temp=z_grid;
    if isa(z_grid,'struct')
        names=fieldnames(z_grid);
        z_grid_temp=z_grid.(names{ii});
    end
    pi_z_temp=pi_z;
    if isa(pi_z,'struct')
        names=fieldnames(pi_z);
        pi_z_temp=pi_z.(names{ii});
    end
    Policy_temp=Policy;
    if isa(Policy,'struct')
        names=fieldnames(Policy);
        Policy_temp=Policy.(names{ii});
    end
    if isstruct(InitialDist)
        InitialDist_temp=InitialDist.(Names_i{ii});
    else
        InitialDist_temp=InitialDist; % Any dependence on permanent type must be done as a structure
    end
    if isfield(simoptions,'ExogShockFnParamNames')==1
        disp('WARNING: ExogShockFn not yet implemented for PType (in SimPanelValues_FHorz_PType_Case1)')
    end
    
    % Parameters are allowed to be given as structure, or as vector/matrix
    % (in terms of their dependence on fixed type). So go through each of
    % these in turn.
    Parameters_temp=Parameters;
    FullParamNames=fieldnames(Parameters);
    nFields=length(FullParamNames);
    for kField=1:nFields
        if isa(Parameters.(FullParamNames{kField}), 'struct') % Check for permanent type in structure form
            names=fieldnames(Parameters.(FullParamNames{kField}));
            Parameters_temp.(FullParamNames{kField})=Parameters.(FullParamNames{kField}).(names{ii});
        elseif sum(size(Parameters.(FullParamNames{kField}))==N_i)==1 % Check for permanent type in vector/matrix form.
            temp=Parameters.(FullParamNames{kField});
            [~,ptypedim]=max(size(Parameters.(FullParamNames{kField}))==N_i); % Parameters as vector/matrix can be at most two dimensional, figure out which relates to PType.
            if ptypedim==1
                Parameters_temp.(FullParamNames{kField})=temp(ii,:);
            elseif ptypedim==2
                Parameters_temp.(FullParamNames{kField})=temp(:,ii);
            end
        end
    end
    
    Parameters_temp

    SimLifeCycleProfiles.(names{ii})=SimLifeCycleProfiles_FHorz_Case1(InitialDist_temp,Policy_temp,FnsToEvaluate,[],Parameters_temp,n_d,n_a,n_z,N_j,d_grid_temp,a_grid_temp,z_grid_temp,pi_z_temp, simoptions_ii);   
    
end


