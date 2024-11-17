function income = Mod_Income(aprime,a,zn,zh,theta,agej,Jr,kappa_j,varrho,w,pension)
% aprime: next-period assets
% a:      current assets
% zn:     labor productivity shock
% zh:     health shock (0=bad, 1=good)
% theta:  permanent type (education)

% If zh=0 (bad health), there is an income penalty. Note that varrho<1
varrho_bad = varrho*(zh==0)+(zh==1);

income = (agej<Jr)*w*kappa_j*zn*theta*varrho_bad+(agej>=Jr)*pension;

end %end function
