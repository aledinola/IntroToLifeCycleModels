function F=AleModel24_ReturnFn(h,aprime,a,z1,z2,e,theta_i,kappa_j,w,sigma,psi,eta,agej,Jr,pen,r)
% The first four are the 'always required' decision variables, next period
% endogenous states, this period endogenous states, exogenous states
% After that we need all the parameters the return function uses, it
% doesn't matter what order we put them here.

F=-Inf;
% Productivity cut is equal to 50% if bad health, to nothing if good health

if agej<Jr % If working age
    c=w*kappa_j*theta_i*z1*z2*e*h+(1+r)*a-aprime; % Add z here
else % Retirement
    c=pen+(1+r)*a-aprime;
end

if c>0
    F=(c^(1-sigma))/(1-sigma) -psi*(h^(1+eta))/(1+eta); % The utility function
end

end %end function
