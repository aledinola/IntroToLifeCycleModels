function [mu_cpu,Policy_cpu] = reshape_VandPolicy(StatDist,Policy,n_a,n_z,n_e,N_i,N_j)
% Reshape stationary distribution and policy functions
% Assumptions: 
% only 2 permanent types (low and high), 
% 2 z shocks, 
if iscell(N_i)
    N_i = numel(N_i);
end

mysize = [n_a,n_z(1),n_z(2),n_e]; %without perm type and without age

%% Gather and reshape policy functions
Policy_1 = squeeze(gather(Policy.low));
Policy_2 = squeeze(gather(Policy.high));

if ~isequal(size(Policy_1),[2,mysize,N_j])
    error('Policy has wrong size')
end

Policy_cpu = zeros(2,n_a,n_z(1),n_z(2),n_e,N_i,N_j);
for j=1:N_j
            %  2,a,z1,z2,e
    Policy_cpu(:,:,:,:,:,1,j) = Policy_1(:,:,:,:,:,j);
    Policy_cpu(:,:,:,:,:,2,j) = Policy_2(:,:,:,:,:,j);
end

%% Gather and reshape distribution
StatDist_1 = gather(StatDist.low); % (a,z1,z2,e,j)
StatDist_2 = gather(StatDist.high);
mu_cpu = zeros(n_a,n_z(1),n_z(2),n_e,N_i,N_j);
for j=1:N_j
    mu_cpu(:,:,:,:,1,j) = StatDist_1(:,:,:,:,j)*StatDist.ptweights(1);
    mu_cpu(:,:,:,:,2,j) = StatDist_2(:,:,:,:,j)*StatDist.ptweights(2);
end

end %end function