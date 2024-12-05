function [relpop,relz,g] = mylorenz(val,pop)

% Add something to deal with zero mass points, NaN, etc.
assert(numel(pop) == numel(val), ...
    'gini expects two equally long vectors (%d ~= %d).', ...
    size(pop,1),size(val,1))

pop = [0;pop(:)]; val = [0;val(:)];     % pre-append a zero

isok = all(~isnan([pop,val]'))';        % filter out NaNs
pop = pop(isok); val = val(isok);

z = val.*pop;
[~,ord] = sort(val);
pop     = pop(ord);     
z       = z(ord);
pop     = cumsum(pop);  
z       = cumsum(z);
relpop  = pop/pop(end); 
relz    = z/z(end);

% From here you can compute the Gini easily...
g = 1 - sum((relz(1:end-1)+relz(2:end)) .* diff(relpop));

end