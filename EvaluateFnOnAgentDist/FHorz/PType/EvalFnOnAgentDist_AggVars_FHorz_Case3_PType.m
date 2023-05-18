function AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case3_PType(StationaryDist, Policy, FnsToEvaluate, Parameters, n_d, n_a, n_z, N_j, Names_i, d_grid, a_grid, z_grid, simoptions)
% Because Case3 policy is same as for Case2 there is no difference in
% EvalFnOnAgentDist between Case2 and Case3. The Case3 command is therefore
% just a front to redirect to Case2. This function exists solely so that
% the user does not have to understand VFI Toolkit so deeply that they know
% Case2 and Case3, while very different for value fn and agent dist, are
% just the same to EvalFnOnAgentDist.

AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case2_PType(StationaryDist, Policy, FnsToEvaluate, Parameters, n_d, n_a, n_z, N_j, Names_i, d_grid, a_grid, z_grid, simoptions);

end

