function F=Mod_ReturnFn(aprime,a,z,theta,agej,Jr,kappa_j,w,r,pension,sigma)

income=(agej<Jr)*w*kappa_j*z*theta+(agej>=Jr)*pension;
cons = (1+r)*a+income-aprime; 

F=-Inf;
if cons>0
    F=(cons^(1-sigma))/(1-sigma) ; % The utility function
end

end %end function
