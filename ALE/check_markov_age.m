function [] = check_markov_age(trans_mat,n_z,J)
% Check that 3 dim trnasition matrix is OK

if ~isequal(size(trans_mat),[n_z,n_z,J])
    error('trans_mat has wrong size')
end
if any(trans_mat<0,"all")
    error('negative elements')
end
if any(trans_mat>1,"all")
    error('elements larger than 1')
end

for jj=1:J
    trans = trans_mat(:,:,jj);
    sum_rows = sum(trans,2);
    if any(abs(sum_rows-1)>1e-8)
        fprintf('j = %d \n',jj)
        error('Row does not some to one for j')
    end
end

end