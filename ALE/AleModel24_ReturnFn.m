function F=AleModel24_ReturnFn(h,aprime,a,z1,z2,e,theta_i,kappa_j,w,sigma,psi,eta,agej,Jr,pen,r,cut)
% The first four are the 'always required' decision variables, next period
% endogenous states, this period endogenous states, exogenous states
% After that we need all the parameters the return function uses, it
% doesn't matter what order we put them here.

F=-Inf;

% prod_loss = 1;
% if z2==0
%     % Bad health
%     prod_loss = cut;
% end

prod_loss = (z2==0)*cut+(z2==1);

if agej<Jr % If working age
    c=w*kappa_j*theta_i*z1*e*prod_loss*h+(1+r)*a-aprime; % Add z here
else % Retirement
    c=pen+(1+r)*a-aprime;
end

if c>0
    F=(c^(1-sigma))/(1-sigma) -psi*(h^(1+eta))/(1+eta); % The utility function
end

end %end function
