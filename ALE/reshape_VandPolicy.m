function [mu,Policy] = reshape_VandPolicy(StatDist,Policy,n_a,n_z,n_e,N_i,N_j)
% Reshape stationary distribution and policy functions
% Assumptions: 
% only 2 permanent types (low and high), 
% 2 z shocks, 
% no d variable
if iscell(N_i)
    N_i = numel(N_i);
end

mysize = [n_a,n_z(1),n_e];

StatDist_1 = gather(StatDist.ptype001);
StatDist_2 = gather(StatDist.ptype002);
Policy_1 = squeeze(gather(Policy.ptype001));
Policy_2 = squeeze(gather(Policy.ptype002));

if ~isequal(size(Policy_1),[2,mysize,N_j])
    error('Policy has wrong size')
end

mu = zeros(n_a,n_z(1),n_e,N_i,N_j);
mu(:,:,:,1,:) = StatDist_1*StatDist.ptweights(1);
mu(:,:,:,2,:) = StatDist_2*StatDist.ptweights(2);
Policy = zeros(2,n_a,n_z(1),n_e,N_i,N_j);
Policy(:,:,:,:,1,:) = Policy_1;
Policy(:,:,:,:,2,:) = Policy_2;


end