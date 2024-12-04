function [] = idea_fun()

n_d = 51;
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

RetMat = reshape(RetMat,[n_d*n_aprime,n_a,n_z]); %(d*a',a,z)

disp('Method 1')
tic
% Initialize output
V1      = zeros(n_a,n_z);
Policy1 = ones(2,n_a,n_z);

for z_c=1:n_z
    EVz = V_next*pi_z(z_c,:)'; %(aprime,1)
    EVz_big = kron(ones(n_d,1),EVz);
    for a_c=1:n_a
        entireRHS = RetMat(:,a_c,z_c)+beta*EVz_big; %(d*a',1)
        [max_val,max_ind] = max(entireRHS); %scalar
        [d_opt,aprime_opt] = ind2sub([n_d,n_aprime],max_ind);
        V1(a_c,z_c) = max_val;
        Policy1(1,a_c,z_c)=d_opt;
        Policy1(2,a_c,z_c)=aprime_opt;
    end
end
toc

disp('Method 2 (vectorized)')
tic
V2      = zeros(n_a,n_z);
Policy2 = ones(2,n_a,n_z);

for z_c=1:n_z
    EVz = V_next*pi_z(z_c,:)'; %(aprime,1)
    EVz_big = kron(ones(n_d,1),EVz);
    %for a_c=1:n_a
        entireRHS = RetMat(:,:,z_c)+beta*EVz_big; %(d*a',a)
        [max_val,max_ind] = max(entireRHS,[],1); %(1,a)
        [d_opt,aprime_opt] = ind2sub([n_d,n_aprime],max_ind);
        V2(:,z_c) = max_val;
        Policy2(1,:,z_c)=d_opt;
        Policy2(2,:,z_c)=aprime_opt;
    %end
end
toc

err = max(abs(V2-V1),[],"all");
disp(err)
err = max(abs(Policy2-Policy1),[],"all");
disp(err)


end