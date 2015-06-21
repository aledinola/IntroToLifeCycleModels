function Values=ValuesOnSSGrid_Case1(ValuesFn,ValuesFnParams,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,p_val,Parallel)
% IS THERE a way to make this work if the function also depends on pi_z???


if Parallel~=2
    disp('ValuesOnSSGrid_Case1() only works for Parallel==2')
end

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);
N_a=prod(n_a);
N_z=prod(n_z);

ParamCell=cell(length(ValuesFnParams),1);
for ii=1:length(ValuesFnParams)
    ParamCell(ii,1)={ValuesFnParams(ii)};
end


if l_d==0 && l_a==1 && l_z==1
    aprimevals=PolicyValuesPermute(:,:,1);
    avals=a_grid;
    zvals=shiftdim(z_grid,-1);
    Values=arrayfun(ValuesFn, aprimevals, avals, zvals,p_val, ParamCell{:});
elseif l_d==0 && l_a==1 && l_z==2
    aprimevals=PolicyValuesPermute(:,:,:,1);
    avals=a_grid;
    z1vals=shiftdim(z_grid(1:n_z(1)),-1);
    z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-2);
    Values=arrayfun(ValuesFn, aprimevals, avals, z1vals,z2vals,p_val, ParamCell{:});
elseif l_d==0 && l_a==2 && l_z==1
    a1primevals=PolicyValuesPermute(:,:,:,1);
    a2primevals=PolicyValuesPermute(:,:,:,2);
    a1vals=a_grid(1:n_a(1));
    a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-1);
    zvals=shiftdim(z_grid,-2);
    Values=arrayfun(ValuesFn, a1primevals,a2primevals, a1vals,a2vals, zvals,p_val, ParamCell{:});
elseif l_d==0 && l_a==2 && l_z==2
    a1primevals=PolicyValuesPermute(:,:,:,:,1);
    a2primevals=PolicyValuesPermute(:,:,:,:,2);
    a1vals=a_grid(1:n_a(1));
    a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-1);
    z1vals=shiftdim(z_grid(1:n_z(1)),-2);
    z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-3);
    Values=arrayfun(ValuesFn, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,p_val, ParamCell{:});
elseif l_d==1 && l_a==1 && l_z==1
    dvals=PolicyValuesPermute(:,:,1);
    aprimevals=PolicyValuesPermute(:,:,2);
    avals=a_grid;
    zvals=shiftdim(z_grid,-1);
    Values=arrayfun(ValuesFn, dvals, aprimevals, avals, zvals,p_val, ParamCell{:});
elseif l_d==1 && l_a==1 && l_z==2
    dvals=PolicyValuesPermute(:,:,:,1);
    aprimevals=PolicyValuesPermute(:,:,:,2);
    avals=a_grid;
    z1vals=shiftdim(z_grid(1:n_z(1)),-1);
    z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-2);
    Values=arrayfun(ValuesFn, dvals, aprimevals, avals, z1vals,z2vals,p_val, ParamCell{:});
elseif l_d==1 && l_a==2 && l_z==1
    dvals=PolicyValuesPermute(:,:,:,1);
    a1primevals=PolicyValuesPermute(:,:,:,2);
    a2primevals=PolicyValuesPermute(:,:,:,3);
    a1vals=a_grid(1:n_a(1));
    a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-1);
    zvals=shiftdim(z_grid,-2);
    Values=arrayfun(ValuesFn, dvals, a1primevals,a2primevals, a1vals,a2vals, zvals,p_val, ParamCell{:});
elseif l_d==1 && l_a==2 && l_z==2
    dvals=PolicyValuesPermute(:,:,:,:,1);
    a1primevals=PolicyValuesPermute(:,:,:,:,2);
    a2primevals=PolicyValuesPermute(:,:,:,:,3);
    a1vals=a_grid(1:n_a(1));
    a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-1);
    z1vals=shiftdim(z_grid(1:n_z(1)),-2);
    z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-3);
    Values=arrayfun(ValuesFn, dvals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,p_val, ParamCell{:});
elseif l_d==2 && l_a==1 && l_z==1
    d1vals=PolicyValuesPermute(:,:,1);
    d2vals=PolicyValuesPermute(:,:,2);
    aprimevals=PolicyValuesPermute(:,:,3);
    avals=a_grid;
    zvals=shiftdim(z_grid,-1);
    Values=arrayfun(ValuesFn, d1vals, d2vals, aprimevals, avals, zvals,p_val, ParamCell{:});
elseif l_d==2 && l_a==1 && l_z==2
    d1vals=PolicyValuesPermute(:,:,:,1);
    d2vals=PolicyValuesPermute(:,:,:,2);
    aprimevals=PolicyValuesPermute(:,:,:,3);
    avals=a_grid;
    z1vals=shiftdim(z_grid(1:n_z(1)),-1);
    z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-2);
    Values=arrayfun(ValuesFn, d1vals, d2vals, aprimevals, avals, z1vals,z2vals,p_val, ParamCell{:});
elseif l_d==2 && l_a==2 && l_z==1
    d1vals=PolicyValuesPermute(:,:,:,1);
    d2vals=PolicyValuesPermute(:,:,:,2);
    a1primevals=PolicyValuesPermute(:,:,:,3);
    a2primevals=PolicyValuesPermute(:,:,:,4);
    a1vals=a_grid(1:n_a(1));
    a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-1);
    zvals=shiftdim(z_grid,-2);
    Values=arrayfun(ValuesFn, d1vals, d2vals, a1primevals,a2primevals, a1vals,a2vals, zvals,p_val, ParamCell{:});
elseif l_d==2 && l_a==2 && l_z==2
    d1vals=PolicyValuesPermute(:,:,:,:,1);
    d2vals=PolicyValuesPermute(:,:,:,:,2);
    a1primevals=PolicyValuesPermute(:,:,:,:,3);
    a2primevals=PolicyValuesPermute(:,:,:,:,4);
    a1vals=a_grid(1:n_a(1));
    a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-1);
    z1vals=shiftdim(z_grid(1:n_z(1)),-2);
    z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-3);
    Values=arrayfun(ValuesFn, d1vals, d2vals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,p_val, ParamCell{:});
end

Values=reshape(Values,[N_a,N_z]);


end