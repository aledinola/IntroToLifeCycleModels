function cons=Mod_Cons(aprime,a,z,theta,agej,Jr,kappa_j,w,r,pension)

income=(agej<Jr)*w*kappa_j*z*theta+(agej>=Jr)*pension;
cons = (1+r)*a+income-aprime; 

end %end function
