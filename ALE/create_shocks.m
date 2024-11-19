function [z_grid_J,pi_z_J,zh_grid_J,pi_zh_J,dist_health] = create_shocks(n_z,N_j,Par)
% PURPOSE
%  This function creates grids and transition matrices for income (n1 points) 
%  and health (n2 points) shocks. Note that health shock depends on age and 
%  on permanent type theta.
% OUTPUTS
%  z_grid_J: Structure with arrays of size [n1+n2,N_j]
%  pi_z_J:   Structure with arrays of size [n1*n2,n1*n2,N_j]

n1 = n_z(1);
n2 = n_z(2);

% Income shock
[zn_grid,pi_zn]=discretizeAR1_Rouwenhorst(0.0,Par.rho,Par.sigma_eps,n1);
zn_grid = exp(zn_grid);

% Health shock, age-dependent and Ptype-dependent {low,high}
% 0=bad health, 1=good health
dist_health = [0.05,0.95]; % 5% in bad health at age j=1
zh_grid_J.low  = repmat([0,1]',1,N_j); % [n2,N_j]
zh_grid_J.high = repmat([0,1]',1,N_j); % [n2,N_j]
% bb: Prob of having bad health tomorrow given that health today is bad
p_bb = linspace(0.6,0.9,N_j)';
% gb: Prob of having bad health tomorrow given that health today is good
% 0.1-0.4 for theta(1) and 0.0-0.2 for theta(2)
p_gb_low  = linspace(0.1,0.4,N_j)';
p_gb_high = linspace(0.0,0.2,N_j)';
pi_zh_J.low  = zeros(n2,n2,N_j);
pi_zh_J.high = zeros(n2,n2,N_j);
for jj=1:N_j
    % -- Low type (no college educ)
    % from bad to bad
    pi_zh_J.low(1,:,jj) = [p_bb(jj),1-p_bb(jj)];
    % from good to bad
    pi_zh_J.low(2,:,jj) = [p_gb_low(jj),1-p_gb_low(jj)];
    % -- High type (college educ)
    % from bad to bad
    pi_zh_J.high(1,:,jj) = [p_bb(jj),1-p_bb(jj)];
    % from good to bad
    pi_zh_J.high(2,:,jj) = [p_gb_high(jj),1-p_gb_high(jj)];
end
disp('Check pi_zh_J.low...')
check_markov_age(pi_zh_J.low,n_z(2),N_j)
disp('Check pi_zh_J.high...')
check_markov_age(pi_zh_J.high,n_z(2),N_j)

% Recall two permanent types: low,high
z_grid_J.low=[zn_grid*ones(1,N_j); zh_grid_J.low];   % array with size [n1+n2,N_j]
z_grid_J.high=[zn_grid*ones(1,N_j); zh_grid_J.high]; % array with size [n1+n2,N_j]
pi_z_J.low=zeros(n1*n2,n1*n2,N_j);  % array with size [n1*n2,n1*n2,N_j]
pi_z_J.high=zeros(n1*n2,n1*n2,N_j); % array with size [n1*n2,n1*n2,N_j]
for jj=1:N_j
   pi_z_J.low(:,:,jj)=kron(pi_zh_J.low(:,:,jj), pi_zn);  % note: kron in reverse order
   pi_z_J.high(:,:,jj)=kron(pi_zh_J.high(:,:,jj), pi_zn);  % note: kron in reverse order
end
disp('Check pi_z_J.low...')
check_markov_age(pi_z_J.low,prod(n_z),N_j)
disp('Check pi_z_J.high...')
check_markov_age(pi_z_J.high,prod(n_z),N_j)

if ~isstruct(z_grid_J)
    error('Output z_grid_J must be a structure')
end
if ~isstruct(pi_z_J)
    error('Output pi_z_J must be a structure')
end

end %end function