function income = Mod_Income(aprime,a,z,theta,agej,Jr,kappa_j,w,pension)

income=(agej<Jr)*w*kappa_j*z*theta+(agej>=Jr)*pension;

end %end function
