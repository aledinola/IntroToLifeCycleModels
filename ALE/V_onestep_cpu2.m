function [V,Policy] = V_onestep_cpu2(RetMat,V_next,pi_z,beta,n_d,n_a,n_z)
% RetMat must be (d*a',a,z) 3-dim array

n_aprime = n_a;

V      = zeros(n_a,n_z);
Policy_lin = zeros(n_a,n_z);

EV = V_next*pi_z'; %(a',z)
for z_c=1:n_z
    EVz_big = repelem(EV(:,z_c),n_d); % (d*a',1)
    for a_c=1:n_a
        entireRHS = RetMat(:,a_c,z_c)+beta*EVz_big; %(d*a',1)
        [max_val,max_ind] = max(entireRHS); %scalar
        V(a_c,z_c) = max_val;
        Policy_lin(a_c,z_c)=max_ind;
    end
end

Policy = zeros(2,n_a,n_z);
[d_opt,aprime_opt] = ind2sub([n_d,n_aprime],Policy_lin);
Policy(1,:,:) = d_opt;
Policy(2,:,:) = aprime_opt;


end %end function