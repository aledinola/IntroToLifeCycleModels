function [V,Policy]=ValueFnIter_Case1_FHorz_SDVz_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
if vfoptions.lowmemory>1
    special_n_a=ones(1,length(n_a));
    a_gridvals=CreateGridvals(n_a,a_grid,1); % The 1 at end indicates want output in form of matrix.
end

%% Get the parameter values that depend on the z_state.
l_SDV_z=length(vfoptions.StateDependentVariables_z);
SDV_z=cell(l_SDV_z,1); % Keep the full SDV_z values
SDV_z_gridvals=zeros(N_z,l_SDV_z,'gpuArray'); % Holds just the SDV_z values for the current age
SDV_z_IndepOfagej=ones(l_SDV_z,1);
FullParamNames=fieldnames(Parameters);
for ii=1:l_SDV_z
    SDVz_Name=vfoptions.StateDependentVariables_z{ii};
    for jj=1:length(FullParamNames)
        if strcmp(SDVz_Name,FullParamNames{jj})
            temp=gpuArray(Parameters.(FullParamNames{jj}));
            SDV_z{ii}=temp;
            SDV_z_gridvals(:,ii)=temp(:,end); % Age N_j, or age independent.
            if size(SDV_z{ii},2)>1
                SDV_z_IndepOfagej(ii)=0;
            end
        end
    end
end
% If using lowmemory>0 then can just pass these state dependent parameter
% values as if they were another z_val (as these will all be done 1 by 1 anyway).
special_n_z=[ones(1,length(n_z)),ones(1,l_SDV_z)]; % (special_n_z is only used when lowmemory>0 so no need to worry about its value otherwise)
if vfoptions.lowmemory==0
    fprintf('ERROR: State Dependent Variables that depend on z (and possibly j) are only permitted for lowmemory>0')
    return
end


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if vfoptions.lowmemory==0
    
    %if vfoptions.returnmatrix==2 % GPU
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_gridvals_J(:,:,N_j), ReturnFnParamsVec);
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,:,N_j)=Vtemp;
    Policy(:,:,N_j)=maxindex;

elseif vfoptions.lowmemory==1
    
    for z_c=1:N_z
        z_val=[z_gridvals_J(z_c,:,N_j),SDV_z_gridvals(z_c,:)];
        ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_grid, a_grid, z_val, ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
        V(:,z_c,N_j)=Vtemp;
        Policy(:,z_c,N_j)=maxindex;
    end
    
elseif vfoptions.lowmemory==2

    for z_c=1:N_z
        z_val=[z_gridvals_J(z_c,:,N_j),SDV_z_gridvals(z_c,:)];
        for a_c=1:N_a
            a_val=a_gridvals(z_c,:);
            ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, special_n_a, special_n_z, d_grid, a_val, z_val, ReturnFnParamsVec);
            %Calc the max and it's index
            [Vtemp,maxindex]=max(ReturnMatrix_az);
            V(a_c,z_c,N_j)=Vtemp;
            Policy(a_c,z_c,N_j)=maxindex;

        end
    end   
    
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    jj=N_j-reverse_j;

    if vfoptions.verbose==1
        sprintf('Finite horizon: %i of %i',reverse_j, N_j)
    end
    
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    if min(SDV_z_IndepOfagej)<1 % If the state dependent variable (depending on z) also depends on age j.
        for ii=1:l_SDV_z
            if SDV_z_IndepOfagej(ii)==0
                temp=SDV_z{ii};
                SDV_z_gridvals(:,ii)=temp(:,jj); % Update SDV_z_gridvals for current age j
            end
        end
    end
    
    VKronNext_j=V(:,:,jj+1);
    
    if lowmemory==0
        
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_gridvals_J(:,:,jj), ReturnFnParamsVec);
        
        for z_c=1:N_z
            ReturnMatrix_z=ReturnMatrix(:,:,z_c);
            
            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z_J(z_c,:,jj));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            entireEV_z=kron(EV_z,ones(N_d,1));
            entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*entireEV_z; %*ones(1,N_a,1);
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,z_c,jj)=Vtemp;
            Policy(:,z_c,jj)=maxindex;
        end
        
    elseif lowmemory==1
        for z_c=1:N_z
            z_val=[z_gridvals_J(z_c,:,jj),SDV_z_gridvals(z_c,:)];
            ReturnMatrix_z=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_grid, a_grid, z_val, ReturnFnParamsVec);
            
            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z_J(z_c,:,jj));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            entireEV_z=kron(EV_z,ones(N_d,1));
            entireRHS_z=ReturnMatrix_z+DiscountFactorParamsVec*entireEV_z; %*ones(1,N_a,1);
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_z,[],1);
            V(:,z_c,jj)=Vtemp;
            Policy(:,z_c,jj)=maxindex;
        end
        
    elseif lowmemory==2
        for z_c=1:N_z
            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=VKronNext_j.*(ones(N_a,1,'gpuArray')*pi_z_J(z_c,:,jj));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            entireEV_z=kron(EV_z,ones(N_d,1));
            
            z_val=[z_gridvals_J(z_c,:,jj),SDV_z_gridvals(z_c,:)];
            for a_c=1:N_a
                a_val=a_gridvals(z_c,:);
                ReturnMatrix_az=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, special_n_a, special_n_z, d_grid, a_val, z_val, ReturnFnParamsVec);
                
                entireRHS_az=ReturnMatrix_az+DiscountFactorParamsVec*entireEV_z;
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS_az);
                V(a_c,z_c,jj)=Vtemp;
                Policy(a_c,z_c,jj)=maxindex;
            end
        end
        
    end
end

Policy=zeros(2,N_a,N_z,N_j,'gpuArray'); %NOTE: this is not actually in Kron form
Policy(1,:,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy(2,:,:,:)=shiftdim(ceil(Policy/N_d),-1);

end