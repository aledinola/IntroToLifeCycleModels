function [ave] = MyCondStats(StatDist,Policy,n_z,z_grid,a_grid,n_a,N_i,N_j,P,FnsToEvaluate)
% Compute conditional statistics over the life-cycle. For example, we
% compute average consumption conditional on health status and permanent
% type, for each age j=1,..,J
% TODO: Add other conditional moments if needed.

% (a,zn,zh,theta,j)
% zh:    health shock
% theta: permanent type
% j:     age

n_zn    = n_z(1);
n_zh    = n_z(2);
n_theta = N_i;

z_grid_J   = gather(z_grid);

[mu,Policy] = reshape_VandPolicy(StatDist,Policy,n_a,n_z,N_i,N_j);

zn_grid_J = z_grid_J(1:n_zn,:);
zh_grid_J = z_grid_J(n_zn+1:n_zn+n_zh,:);

frac_types = zeros(n_zh,n_theta,N_j);

for jj = 1:N_j
    for theta_c=1:n_theta
        for zh_c=1:n_zh
            frac_types(zh_c,theta_c,jj)= sum(mu(:,:,zh_c,theta_c,jj),"all");
        end
    end
end

% Compute values of X on the grid
Values_c = zeros(n_a,n_zn,n_zh,n_theta,N_j);
for jj = 1:N_j
for theta_c=1:n_theta
for zh_c=1:n_zh
for zn_c=1:n_zn
for a_c=1:n_a
    aprime = a_grid(Policy(a_c,zn_c,zh_c,theta_c,jj));
    a = a_grid(a_c);
    zn = zn_grid_J(zn_c,jj);
    zh = zh_grid_J(zh_c,jj);
    theta = P.theta(theta_c);
    Values_c(a_c,zn_c,zh_c,theta_c,jj)=...
        FnsToEvaluate.consumption(aprime,a,zn,zh,theta,P.agej(jj),P.Jr,P.kappa_j(jj),P.varrho,P.w,P.r,P.pension);
end
end
end
end
end
% Compute average of X conditional on (zh,theta,j)
ave_c = zeros(n_zh,n_theta,N_j);
for jj = 1:N_j
    for theta_c=1:n_theta
        for zh_c=1:n_zh
            ave_c(zh_c,theta_c,jj)=sum(Values_c(:,:,zh_c,theta_c,jj).*mu(:,:,zh_c,theta_c,jj),"all")/frac_types(zh_c,theta_c,jj);
        end
    end
end

ave.consumption = ave_c;

end %end function