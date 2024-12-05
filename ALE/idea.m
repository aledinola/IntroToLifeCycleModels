clear
clc
close all

n_d = 3;
n_a = 601;
n_aprime = n_a;
n_z = 7;
beta = 0.98;

rng("default")
% Fake data
RetMat = rand(n_d,n_aprime,n_a,n_z);
V_next = rand(n_aprime,n_z);
pi_z   = rand(n_z,n_z);
pi_z   = pi_z./sum(pi_z,2);

RetMat_reshaped = reshape(RetMat,[n_d*n_aprime,n_a,n_z]); %(d*a',a,z)

%% Method 1
disp('Method 1')
tic
% Initialize output
V1      = zeros(n_a,n_z);
Policy1 = ones(n_a,n_z);

for z_c=1:n_z
    EVz = V_next*pi_z(z_c,:)'; %(a',1)
    %EVz_big = kron(EVz,ones(n_d,1));
    EVz_big = repelem(EVz,n_d); % (d*a',1)
    for a_c=1:n_a
        entireRHS = RetMat_reshaped(:,a_c,z_c)+beta*EVz_big; %(d*a',1)
        [max_val,max_ind] = max(entireRHS); %scalar
        V1(a_c,z_c) = max_val;
        Policy1(a_c,z_c)=max_ind;
    end
end

Policy1_all = zeros(2,n_a,n_z);
[d_opt,aprime_opt] = ind2sub([n_d,n_aprime],Policy1);
Policy1_all(1,:,:) = d_opt;
Policy1_all(2,:,:) = aprime_opt;

toc

%% Method 2 (improved)
disp('Method 2 (improved)')
tic
[V2,Policy2_all] = V_onestep_cpu2(RetMat_reshaped,V_next,pi_z,beta,n_d,n_a,n_z);
toc

%% Method 3 - loop over d
tic
[V3,Policy3_all] = V_onestep_cpu3(RetMat,V_next,pi_z,beta,n_d,n_a,n_z);
toc

err1 = max(abs(V2-V1),[],"all");
disp(err1)
err2 = max(abs(Policy2_all-Policy1_all),[],"all");
disp(err2)
err3 = max(abs(Policy3_all-Policy1_all),[],"all");
disp(err3)
