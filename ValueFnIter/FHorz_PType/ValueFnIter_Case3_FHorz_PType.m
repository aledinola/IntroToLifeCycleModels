function [V, Policy]=ValueFnIter_Case3_FHorz_PType(n_d,n_a,n_z,n_u, N_j,Names_i,d_grid, a_grid, z_grid, u_grid, pi_z, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, vfoptions)

%
% vfoptions.verbose=1 will give feedback
% vfoptions.verboseparams=1 will give further feedback on the param values of each permanent type
%

% N_d=prod(n_d);
% N_a=prod(n_a);
% N_z=prod(n_z);
% N_i=prod(n_i);

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

for ii=1:N_i
        
    % First set up vfoptions
    if exist('vfoptions','var')
        vfoptions_temp=PType_Options(vfoptions,Names_i,ii);
        if ~isfield(vfoptions_temp,'verbose')
            vfoptions_temp.verbose=0;
        end
        if ~isfield(vfoptions_temp,'verboseparams')
            vfoptions_temp.verboseparams=0;
        end
        if ~isfield(vfoptions_temp,'ptypestorecpu')
            vfoptions_temp.ptypestorecpu=1; % GPU memory is limited, so switch solutions to the cpu
        end
    else
        vfoptions_temp.verbose=0;
        vfoptions_temp.verboseparams=0;
        vfoptions_temp.ptypestorecpu=1; % GPU memory is limited, so switch solutions to the cpu
    end 
    
    if vfoptions_temp.verbose==1
        fprintf('Permanent type: %i of %i \n',ii, N_i)
    end
    
    % Go through everything which might be dependent on fixed type (PType)
    % [THIS could be better coded, 'names' are same for all these and just need to be found once outside of ii loop]
    if isa(n_d,'struct')
        n_d_temp=n_d.(Names_i{ii});
    else
        n_d_temp=n_d;
    end
    if isa(n_a,'struct')
        n_a_temp=n_a.(Names_i{ii});
    else
        n_a_temp=n_a;
    end
    if isa(n_z,'struct')
        n_z_temp=n_z.(Names_i{ii});
    else
        n_z_temp=n_z;
    end
    if isa(n_u,'struct')
        n_u_temp=n_u.(Names_i{ii});
    else
        n_u_temp=n_u;
    end
    if isa(N_j,'struct')
        N_j_temp=N_j.(Names_i{ii});
    else
        N_j_temp=N_j;
    end
    if isa(d_grid,'struct')
        d_grid_temp=d_grid.(Names_i{ii});
    else
        d_grid_temp=d_grid;
    end
    if isa(a_grid,'struct')
        a_grid_temp=a_grid.(Names_i{ii});
    else
        a_grid_temp=a_grid;
    end
    if isa(z_grid,'struct')
        z_grid_temp=z_grid.(Names_i{ii});
    else
        z_grid_temp=z_grid;
    end
    if isa(u_grid,'struct')
        u_grid_temp=u_grid.(Names_i{ii});
    else
        u_grid_temp=u_grid;
    end
    if isa(pi_z,'struct')
        pi_z_temp=pi_z.(Names_i{ii});
    else
        pi_z_temp=pi_z;
    end
    if isa(pi_u,'struct')
        pi_u_temp=pi_u.(Names_i{ii});
    else
        pi_u_temp=pi_u;
    end
    if isa(ReturnFn,'struct')
        ReturnFn_temp=ReturnFn.(Names_i{ii});
    else
        ReturnFn_temp=ReturnFn;
    end
    if isa(aprimeFn,'struct')
        aprimeFn_temp=aprimeFn.(Names_i{ii});
    else
        aprimeFn_temp=aprimeFn;
    end

    % Parameters are allowed to be given as structure, or as vector/matrix
    % (in terms of their dependence on fixed type). So go through each of
    % these in term.
    Parameters_temp=Parameters;
    FullParamNames=fieldnames(Parameters);
    nFields=length(FullParamNames);
    for kField=1:nFields
        if isa(Parameters.(FullParamNames{kField}), 'struct') % Check for permanent type in structure form
            names=fieldnames(Parameters.(FullParamNames{kField}));
            for jj=1:length(names)
                if strcmp(names{jj},Names_i{ii})
                    Parameters_temp.(FullParamNames{kField})=Parameters.(FullParamNames{kField}).(names{jj});
                end
            end
        elseif any(size(Parameters.(FullParamNames{kField}))==N_i) % Check for permanent type in vector/matrix form.
            temp=Parameters.(FullParamNames{kField});
            [~,ptypedim]=max(size(Parameters.(FullParamNames{kField}))==N_i); % Parameters as vector/matrix can be at most two dimensional, figure out which relates to PType.
            if ptypedim==1
                Parameters_temp.(FullParamNames{kField})=temp(ii,:);
            elseif ptypedim==2
                Parameters_temp.(FullParamNames{kField})=temp(:,ii);
            end
        end
    end
    DiscountFactorParamNames_temp=DiscountFactorParamNames;
    if isa(DiscountFactorParamNames,'struct')
        names=fieldnames(DiscountFactorParamNames);
        for jj=1:length(names)
            if strcmp(names{jj},Names_i{ii})
                DiscountFactorParamNames_temp=DiscountFactorParamNames.(names{jj});
            end
        end
    end
    
    if vfoptions_temp.verboseparams==1
        sprintf('Parameter values for the current permanent type')
        Parameters_temp
    end
    
    if isfinite(N_j_temp)
        [V_ii, Policy_ii]=ValueFnIter_Case3_FHorz(n_d_temp,n_a_temp,n_z_temp,n_u_temp,N_j_temp,d_grid_temp, a_grid_temp, z_grid_temp, u_grid_temp, pi_z_temp, pi_u_temp, ReturnFn_temp, aprimeFn_temp, Parameters_temp, DiscountFactorParamNames_temp, [], vfoptions_temp);
    else % PType actually allows for infinite horizon as well
        error('Infinite horizon not currently supported for Case3')
%         [V_ii, Policy_ii]=ValueFnIter_Case1(n_d_temp,n_a_temp,n_z_temp,d_grid_temp, a_grid_temp, z_grid_temp, pi_z_temp, ReturnFn_temp, Parameters_temp, DiscountFactorParamNames_temp, [], vfoptions_temp);
    end
    
    if vfoptions_temp.ptypestorecpu==1
        V.(Names_i{ii})=gather(V_ii);
        Policy.(Names_i{ii})=gather(Policy_ii);
    else
        V.(Names_i{ii})=V_ii;
        Policy.(Names_i{ii})=Policy_ii;
    end
        
    clear V_ii Policy_ii

end


end