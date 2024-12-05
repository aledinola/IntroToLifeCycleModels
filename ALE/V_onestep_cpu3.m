function [V,Policy] = V_onestep_cpu3(RetMat,V_next,pi_z,beta,n_d,n_a,n_z)
% RetMat must be (d,a',a,z) 4-dim array
%[n_a,n_z] = size(V_next);

V_d =zeros(n_a,n_z,n_d);
Policy_d =zeros(n_a,n_z,n_d);

for d_c=1:n_d
    for z_c=1:n_z
        prob = pi_z(z_c,:)'; % (z',1)
        EV_z = V_next*prob; % (a',1)
        RetMat2 = reshape(RetMat(d_c,:,:,z_c),n_a,n_a);
        entireRHS = RetMat2+beta*EV_z; %(a',a)
        [max_val,max_ind] = max(entireRHS,[],1); %(1,a)
        V_d(:,z_c,d_c)= max_val;
        Policy_d(:,z_c,d_c)= max_ind;
    end %end z
end %end a

[V,d_max] = max(V_d,[],3);
Policy = zeros(2,n_a,n_z);
Policy(1,:,:) = d_max;
for z_c = 1:n_z
    for a_c=1:n_a
        Policy(2,a_c,z_c) = Policy_d(a_c,z_c,d_max(a_c,z_c));
    end
end

end %end function