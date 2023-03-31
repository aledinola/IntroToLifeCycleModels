function [V,Policy]=ValueFnIter_Case2_11_FHorz_Par2_raw(n_d,n_a,n_z,N_j, d_grid, a_grid, z_grid, pi_z,Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

V=zeros(N_a,N_z,N_j,'gpuArray');
Policy=zeros(N_a,N_z,N_j,'gpuArray'); %indexes the optimal choice for d given rest of dimensions a,z

%%

eval('fieldexists_pi_z_J=1;vfoptions.pi_z_J;','fieldexists_pi_z_J=0;')
eval('fieldexists_ExogShockFn=1;vfoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;vfoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')

if vfoptions.lowmemory>0 || Case2_Type==11
    special_n_z=ones(1,length(n_z));
    z_gridvals=CreateGridvals(n_z,z_grid,1); % The 1 at end indicates want output in form of matrix.
end
if vfoptions.lowmemory>1
    special_n_a=ones(1,length(n_a));
    a_gridvals=CreateGridvals(n_a,a_grid,1); % The 1 at end indicates want output in form of matrix.
end

%%

%% j=N_j

if vfoptions.verbose==1
    sprintf('Age j is currently %i \n',N_j)
end

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


if vfoptions.lowmemory==0
    ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParamsVec);
    %Calc the max and it's index
    [Vtemp,maxindex]=max(ReturnMatrix,[],1);
    V(:,:,N_j)=Vtemp;
    Policy(:,:,N_j)=maxindex;
    
elseif vfoptions.lowmemory==1
    for z_c=1:N_z
        z_val=z_gridvals(z_c,:);
        ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_grid, a_grid, z_val, ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix_z,[],1);
        V(:,z_c,N_j)=Vtemp;
        Policy(:,z_c,N_j)=maxindex;
    end
    
elseif vfoptions.lowmemory==2
    for a_c=1:N_a
        a_val=a_gridvals(a_c,:);
        ReturnMatrix_a=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, special_n_a, n_z, d_grid, a_val, z_grid, ReturnFnParamsVec);
        %Calc the max and it's index
        [Vtemp,maxindex]=max(ReturnMatrix_a,[],1);
        V(a_c,:,N_j)=Vtemp;
        Policy(a_c,:,N_j)=maxindex;
    end
    
end

% Case2_Type==11 % phi_a'(d,a,z')

if vfoptions.phiaprimedependsonage==0
    PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames);
    if vfoptions.lowmemory==0
        Phi_aprimeMatrix=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec);
    end
end

for reverse_j=1:N_j-1
    jj=N_j-reverse_j;
    
    if vfoptions.verbose==1
        sprintf('Age j is currently %i \n',jj)
    end
    
    % Create a vector containing all the return function parameters (in order)
    ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames,jj);
    DiscountFactorParamsVec=CreateVectorFromParams(Parameters, DiscountFactorParamNames,jj);
    
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
    
    Vnextj=V(:,:,jj+1);
    
    if vfoptions.lowmemory==0 % CAN PROBABLY IMPROVE THIS (lowmemory=0 is just doing same as lowmemory=1 for Case2_Type=11)
        % Current Case2_Type=11: phi_a'(d,a,z')
        if vfoptions.phiaprimedependsonage==1
            PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
        end
        Phi_aprimeMatrix_z=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec); %z_grid as doing all of zprime (note that it is independent of z as Case2_Type=11)
        Phi_aprimeMatrix_z=reshape(Phi_aprimeMatrix_z,[N_d*N_a*N_z,1]);
        aaaVnextj=kron(ones(N_d,1),Vnextj);
        zprime_ToMatchPhi=kron((1:1:N_z)',ones(N_d*N_a,1));
        for z_c=1:N_z
            z_val=z_gridvals(z_c,:); % Value of z (not of z')
            aaa=kron(pi_z(z_c,:),ones(N_d*N_a,1,'gpuArray'));

            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_grid, a_grid, z_val, ReturnFnParamsVec);
            
            EV_z=aaaVnextj(Phi_aprimeMatrix_z+(N_d*N_a)*(zprime_ToMatchPhi-1)); %*aaa;
            
            EV_z=reshape(EV_z,[N_d*N_a,N_z]);
            EV_z=EV_z.*aaa;
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=reshape(sum(EV_z,2),[N_d,N_a]);
            
            entireRHS=ReturnMatrix_z+DiscountFactorParamsVec*EV_z; %d by a (by z)
            
            %calculate in order, the maximizing aprime indexes
            [V(:,z_c,jj),Policy(:,z_c,jj)]=max(entireRHS,[],1);
        end
    elseif vfoptions.lowmemory==1
        % Current Case2_Type=11: phi_a'(d,a,z')
        if vfoptions.phiaprimedependsonage==1
            PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
        end
        Phi_aprimeMatrix_z=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, n_a, n_z, d_grid, a_grid, z_grid,PhiaprimeParamsVec); %z_grid as doing all of zprime (note that it is independent of z as Case2_Type=11)
        Phi_aprimeMatrix_z=reshape(Phi_aprimeMatrix_z,[N_d*N_a*N_z,1]);
        aaaVnextj=kron(ones(N_d,1),Vnextj);
        zprime_ToMatchPhi=kron((1:1:N_z)',ones(N_d*N_a,1));
        for z_c=1:N_z
            z_val=z_gridvals(z_c,:); % Value of z (not of z')
            aaa=kron(pi_z(z_c,:),ones(N_d*N_a,1,'gpuArray'));
            
            ReturnMatrix_z=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, special_n_z, d_grid, a_grid, z_val, ReturnFnParamsVec);
            EV_z=aaaVnextj(Phi_aprimeMatrix_z+(N_d*N_a)*(zprime_ToMatchPhi-1)); %*aaa;
            
            EV_z=reshape(EV_z,[N_d*N_a,N_z]);
            EV_z=EV_z.*aaa;
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=reshape(sum(EV_z,2),[N_d,N_a]);
            
            entireRHS=ReturnMatrix_z+DiscountFactorParamsVec*EV_z; %d by a (by z)
            
            %calculate in order, the maximizing aprime indexes
            [V(:,z_c,jj),Policy(:,z_c,jj)]=max(entireRHS,[],1);
            
        end
    elseif vfoptions.lowmemory==2
        for a_c=1:N_a
            if vfoptions.phiaprimedependsonage==1
                PhiaprimeParamsVec=CreateVectorFromParams(Parameters, PhiaprimeParamNames,jj);
            end
            Phi_aprimeMatrix_a=CreatePhiaprimeMatrix_Case2_Disc_Par2(Phi_aprime, Case2_Type, n_d, special_n_a, n_z, d_grid, a_val, z_grid,PhiaprimeParamsVec);
            ReturnMatrix_a=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, special_n_a,n_z, d_grid, a_val, z_grid, ReturnFnParamsVec);
            for z_c=1:N_z
                RHSpart2=zeros(N_d,1);
                for zprime_c=1:N_z
                    if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
                        for d_c=1:N_d
                            RHSpart2(d_c)=RHSpart2(d_c)+Vnextj(Phi_aprimeMatrix_a(d_c,1,z_c,zprime_c),zprime_c)*pi_z(z_c,zprime_c);
                        end
                    end
                end
                entireRHS=ReturnMatrix_a(:,1,z_c)+DiscountFactorParamsVec*RHSpart2; %aprime by 1
                
                %calculate in order, the maximizing aprime indexes
                [V(a_c,z_c,jj),Policy(a_c,z_c,jj)]=max(entireRHS,[],1);
            end
        end
    end
end



end