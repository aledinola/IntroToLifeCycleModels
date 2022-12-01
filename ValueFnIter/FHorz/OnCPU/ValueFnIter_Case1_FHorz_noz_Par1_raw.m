function [V,Policy2]=ValueFnIter_Case1_FHorz_noz_Par1_raw(n_d,n_a,N_j, d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);

V=zeros(N_a,N_j);
Policy=zeros(N_a,N_j); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

%%
a_grid=gather(a_grid);

if vfoptions.lowmemory>0
    special_n_a=ones(1,length(n_a));
    a_gridvals=CreateGridvals(n_a,a_grid,1); % The 1 at end indicates want output in form of matrix.
end


%% j=N_j

% Create a vector containing all the return function parameters (in order)
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,N_j);

if vfoptions.lowmemory==0
    
    %if vfoptions.returnmatrix==2 % GPU
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz(ReturnFn, n_d, n_a, d_grid, a_grid, vfoptions.parallel,ReturnFnParamsVec,0);
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,N_j)=Vtemp;
    Policy(:,N_j)=maxindex;

elseif vfoptions.lowmemory==1 || vfoptions.lowmemory==2

    parfor a_c=1:N_a
        a_val=a_gridvals(a_c,:);
        ReturnMatrix_a=CreateReturnFnMatrix_Case1_Disc_noz(ReturnFn, n_d, special_n_a, d_grid, a_val, 0,ReturnFnParamsVec,0);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix_a);
        V(a_c,N_j)=Vtemp;
        Policy(a_c,N_j)=maxindex;        
    end
    
end

%% Iterate backwards through j.
for reverse_j=1:N_j-1
    j=N_j-reverse_j;

    if vfoptions.verbose==1
        fprintf('Finite horizon: %i of %i \n',j, N_j)
    end
    
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,j);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,j);
    DiscountFactorParamsVec=prod(DiscountFactorParamsVec);
    
    VKronNext_j=V(:,:,j+1);
    
    if vfoptions.lowmemory==0
        
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_noz(ReturnFn, n_d, n_a, d_grid, a_grid, vfoptions.parallel, ReturnFnParamsVec,0);
    
        entireEV=kron(VKronNext_j,ones(N_d,1));
        entireRHS=ReturnMatrix+DiscountFactorParamsVec*entireEV*ones(1,N_a,1);
    
        %Calc the max and it's index
        [Vtemp,maxindex]=max(entireRHS,[],1);
        V(:,j)=Vtemp;
        Policy(:,j)=maxindex;
    
    elseif vfoptions.lowmemory==1 || vfoptions.lowmemory==2
        
        entireEV=kron(VKronNext_j,ones(N_d,1));
        
        parfor a_c=1:N_a
            a_val=a_gridvals(a_c,:);
            ReturnMatrix_a=CreateReturnFnMatrix_Case1_Disc(ReturnFn, n_d, special_n_a, d_grid, a_val, 0,ReturnFnParamsVec,0);
            
            entireRHS_a=ReturnMatrix_a+DiscountFactorParamsVec*entireEV;
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS_a);
            V(a_c,j)=Vtemp;
            Policy(a_c,j)=maxindex;
        end
    end
    
end

%%
Policy2=zeros(2,N_a,N_j); %NOTE: this is not actually in Kron form
Policy2(1,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
Policy2(2,:,:)=shiftdim(ceil(Policy/N_d),-1);

end