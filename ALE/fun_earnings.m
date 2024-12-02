function F=fun_earnings(h,aprime,a,z1,z2,e,theta_i,kappa_j,w,agej,Jr,pen,cut)
% The first four are the 'always required' decision variables, next period
% endogenous states, this period endogenous states, exogenous states
% After that we need all the parameters the return function uses, it
% doesn't matter what order we put them here.

F=-Inf;

prod_loss = 1;
if z2==0
    % Bad health
    prod_loss = cut;
end

if agej<Jr % If working age
    F=w*kappa_j*theta_i*z1*e*prod_loss*h; % Add z here
else % Retirement
    F=pen;
end

end %end function
