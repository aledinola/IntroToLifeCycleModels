function [StationaryDistKron]=StationaryDist_Case1_Iteration_EntryExit_raw(StationaryDistKron,Parameters,EntryExitParamNames,PolicyIndexesKron,N_d,N_a,N_z,pi_z,simoptions)
% Will treat the agents as being on a continuum of mass 1, and then keep
% track of actual mass using StationaryDistKron.mass.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel

% Get the Entry-Exit parameters out of Parameters.
CondlProbOfSurvival=Parameters.(EntryExitParamNames.CondlProbOfSurvival{1});
DistOfNewAgents=Parameters.(EntryExitParamNames.DistOfNewAgents{1});
MassOfNewAgents=Parameters.(EntryExitParamNames.MassOfNewAgents{1});
% StationaryDistKron.mass
% StationaryDistKron.pdf

% simoptions.DistOfNewAgents=kron(pistar_tau,pistar_s); % Note: these should be in 'reverse order'
% simoptions.CondlProbOfSurvivalParamNames={'lambda'};
% simoptions.CondlProbOfSurvival=@(lambda) 1-lambda;
% simoptions.MassOfNewAgents

%% Get the entry and exit variables into the appropriate form.
% % Check whether CondlProbOfSurvival is a function, matrix, or scalar, and act accordingly.
% if isa(simoptions.CondlProbOfSurvival,'function_handle')
%     % Implicitly assume there is only one parameter in simoptions.CondlProbOfSurvivalParamNames
%     Values=StateDependentParam_az(Params,simoptions.CondlProbOfSurvivalParamNames{:},DependenceVec,n_a,n_z,1,1,simoptions.parallel);
%     CondlProbOfSurvival=arrayfun(simoptions.CondlProbOfSurvival, Values);
%     CondlProbOfSurvival=reshape(CondlProbOfSurvival,[N_a*N_z,1]);
% Check whether CondlProbOfSurvival is a matrix, or scalar, and act accordingly.
if isscalar(gather(CondlProbOfSurvival))
    % No need to do anything
elseif isa(gather(CondlProbOfSurvival),'numeric')
    CondlProbOfSurvival=reshape(CondlProbOfSurvival,[N_a*N_z,1]);
else % Does not appear to have been inputted correctly
    fprintf('ERROR: CondlProbOfSurvival parameter does not appear to have been inputted with correct format \n')
    dbstack
    return
end
% Move these to where they need to be.
if simoptions.parallel==2 % On GPU
    DistOfNewAgentsKron=reshape(gpuArray(DistOfNewAgents),[N_a*N_z,1]);
    CondlProbOfSurvival=gpuArray(CondlProbOfSurvival);
elseif simoptions.parallel<2 % On CPU
    DistOfNewAgentsKron=reshape(gather(DistOfNewAgents),[N_a*N_z,1]);    
    CondlProbOfSurvival=gather(CondlProbOfSurvival);
elseif simoptions.parallel==3 % On CPU, sparse matrix
    DistOfNewAgentsKron=reshape(sparse(gather(DistOfNewAgents)),[N_a*N_z,1]);
    CondlProbOfSurvival=sparse(gather(CondlProbOfSurvival));
end

% Note: CondlProbOfSurvival is [N_a*N_z,1] because it will multiply Ptranspose.

%% First, create Ptranspose
% First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
% (Actually I create it's transpose, as that is what will be used repeatedly later.)

if N_d==0 %length(n_d)==1 && n_d(1)==0
    optaprime=reshape(PolicyIndexesKron,[1,N_a*N_z]);
else
    optaprime=reshape(PolicyIndexesKron(2,:,:),[1,N_a*N_z]);
end
% % if simoptions.endogenousexit==1    
% %     % This causes error with gpu as end up with some indexes being non-unique when doing "Ptranspose(optaprime+N_a*(0:1:N_a*N_z-1))=1;" and gpu cannot do non-unique indexes.
% %     % For this reason it has to be treated seperately below.
% %     
% %     optaprime=optaprime+(1-CondlProbOfSurvival'); % endogenous exit means that CondlProbOfSurvival will be 1-ExitPolicy
% %     % This will make all those who 'exit' instead move to first point on
% %     % 'grid on a'. Since as part of it's creation Ptranspose then gets multiplied by the
% %     % CondlProbOfSurvival these agents will all 'die' anyway.
% %     % It is done as otherwise the optaprime policy is being stored as
% %     % 'zero' for those who exit, and this causes an error when trying to
% %     % use optaprime as an index.
% %     % (Need to use transpose of CondlProbOfSurvival because it is being
% %     % kept in the 'transposed' form as usually is used to multiply Ptranspose.)
% %     
% %     % Note: not an issue when using simoptions.endogenousexit==2
% % end

if simoptions.endogenousexit==0
    if simoptions.parallel<2
        Ptranspose=zeros(N_a,N_a*N_z);
        Ptranspose(optaprime+N_a*(0:1:N_a*N_z-1))=1;
        if isscalar(CondlProbOfSurvival) % Put CondlProbOfSurvival where it seems likely to involve the least extra multiplication operations (so hopefully fastest).
            Ptranspose=(kron(pi_z',ones(N_a,N_a))).*(kron(CondlProbOfSurvival*ones(N_z,1),Ptranspose));
        else
            Ptranspose=(kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose); % The order of operations in this line is important, namely multiply the Ptranspose by the survival prob before the muliplication by pi_z
        end
    elseif simoptions.parallel==2 % Using the GPU
        Ptranspose=zeros(N_a,N_a*N_z,'gpuArray');
        Ptranspose(optaprime+N_a*(0:1:N_a*N_z-1))=1;
        if isscalar(CondlProbOfSurvival) % Put CondlProbOfSurvival where it seems likely to involve the least extra multiplication operations (so hopefully fastest).
            Ptranspose=(kron(pi_z',ones(N_a,N_a,'gpuArray'))).*(kron(CondlProbOfSurvival*ones(N_z,1,'gpuArray'),Ptranspose));
        else
            Ptranspose=(kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose); % The order of operations in this line is important, namely multiply the Ptranspose by the survival prob before the muliplication by pi_z
        end
    elseif simoptions.parallel>2
        Ptranspose=sparse(N_a,N_a*N_z);
        Ptranspose(optaprime+N_a*(0:1:N_a*N_z-1))=1;
        if isscalar(CondlProbOfSurvival) % Put CondlProbOfSurvival where it seems likely to involve the least extra multiplication operations (so hopefully fastest).
            Ptranspose=(kron(pi_z',ones(N_a,N_a))).*(kron(CondlProbOfSurvival*ones(N_z,1),Ptranspose));
        else
            Ptranspose=(kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose); % The order of operations in this line is important, namely multiply the Ptranspose by the survival prob before the muliplication by pi_z
        end
    end
elseif simoptions.endogenousexit==1
    if simoptions.parallel<2
        Ptranspose=zeros(N_a,N_a*N_z);
        temp=optaprime+N_a*(0:1:N_a*N_z-1);
        temp=temp(optaprime>0); % temp is just optaprime conditional on staying
        Ptranspose(temp)=1;
        if isscalar(CondlProbOfSurvival) % Put CondlProbOfSurvival where it seems likely to involve the least extra multiplication operations (so hopefully fastest).
            Ptranspose=kron(pi_z',ones(N_a,N_a)).*kron(CondlProbOfSurvival*ones(N_z,1),Ptranspose);
        else
            Ptranspose=kron(pi_z',ones(N_a,N_a)).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose); % The order of operations in this line is important, namely multiply the Ptranspose by the survival prob before the muliplication by pi_z
        end
    elseif simoptions.parallel==2 % Using the GPU
        Ptranspose=zeros(N_a,N_a*N_z,'gpuArray');
        temp=optaprime+N_a*(gpuArray(0:1:N_a*N_z-1));
        temp=temp(optaprime>0); % temp is just optaprime conditional on staying
        Ptranspose(temp)=1;
        if isscalar(CondlProbOfSurvival) % Put CondlProbOfSurvival where it seems likely to involve the least extra multiplication operations (so hopefully fastest).
            Ptranspose=kron(pi_z',ones(N_a,N_a,'gpuArray')).*kron(CondlProbOfSurvival*ones(N_z,1,'gpuArray'),Ptranspose);
        else
            Ptranspose=kron(pi_z',ones(N_a,N_a)).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose); % The order of operations in this line is important, namely multiply the Ptranspose by the survival prob before the muliplication by pi_z
        end
    elseif simoptions.parallel>2
        Ptranspose=sparse(N_a,N_a*N_z);
        temp=optaprime+N_a*(0:1:N_a*N_z-1);
        temp=temp(optaprime>0); % temp is just optaprime conditional on staying
        Ptranspose(temp)=1;
        if isscalar(CondlProbOfSurvival) % Put CondlProbOfSurvival where it seems likely to involve the least extra multiplication operations (so hopefully fastest).
            Ptranspose=(kron(pi_z',ones(N_a,N_a))).*(kron(CondlProbOfSurvival*ones(N_z,1),Ptranspose));
        else
            Ptranspose=(kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose); % The order of operations in this line is important, namely multiply the Ptranspose by the survival prob before the muliplication by pi_z
        end
    end
elseif simoptions.endogenousexit==2
    exitprobabilities=CreateVectorFromParams(Parameters, simoptions.exitprobabilities);
    exitprobs=[1-sum(exitprobabilities),exitprobabilities];
%     exitprob=simoptions.exitprobabilities;
    % Mixed exit (endogenous and exogenous), so we know that CondlProbOfSurvival=reshape(CondlProbOfSurvival,[N_a*N_z,1]);
    if simoptions.parallel<2
        Ptranspose=zeros(N_a,N_a*N_z);
        Ptranspose(optaprime+N_a*(0:1:N_a*N_z-1))=1;
%         Ptranspose1=(kron(pi_z',ones(N_a,N_a))).*(kron(exitprob(1)*ones(N_z,1),Ptranspose)); % No exit, and remove exog exit
%         Ptranspose2=(kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose); % The order of operations in this line is important, namely multiply the Ptranspose by the survival prob before the muliplication by pi_z
%         Ptranspose=Ptranspose1+exitprob(2)*Ptranspose2; % Add the appropriate for endogenous exit
        % Following line does (in one line) what the above three commented
        % out lines do (doing it in one presumably reduces memory usage of Ptranspose1 and Ptranspose2)
        Ptranspose=((kron(pi_z',ones(N_a,N_a))).*(kron(exitprobs(1)*ones(N_z,1),Ptranspose)))+exitprobs(2)*((kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose)); % Add the appropriate for endogenous exit
    elseif simoptions.parallel==2 % Using the GPU
        exitprobs=gpuArray(exitprobs);
        Ptranspose=zeros(N_a,N_a*N_z,'gpuArray');
        Ptranspose(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=1;
%         Ptranspose1=(kron(pi_z',ones(N_a,N_a,'gpuArray'))).*(kron(exitprob(1)*ones(N_z,1,'gpuArray'),Ptranspose)); % No exit, and remove exog exit
%         Ptranspose2=(kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose); % The order of operations in this line is important, namely multiply the Ptranspose by the survival prob before the muliplication by pi_z
%         Ptranspose=Ptranspose1+exitprob(2)*Ptranspose2; % Add the appropriate for endogenous exit
        Ptranspose=((kron(pi_z',ones(N_a,N_a,'gpuArray'))).*(kron(exitprobs(1)*ones(N_z,1,'gpuArray'),Ptranspose)))+exitprobs(2)*((kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose)); % Add the appropriate for endogenous exit
    elseif simoptions.parallel>2
        Ptranspose=sparse(N_a,N_a*N_z);
        Ptranspose(optaprime+N_a*(0:1:N_a*N_z-1))=1;
%         Ptranspose1=(kron(pi_z',ones(N_a,N_a))).*(kron(exitprob(1)*ones(N_z,1),Ptranspose)); % No exit, and remove exog exit
%         Ptranspose2=(kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose); % The order of operations in this line is important, namely multiply the Ptranspose by the survival prob before the muliplication by pi_z
%         Ptranspose=Ptranspose1+exitprob(2)*Ptranspose2; % Add the appropriate for endogenous exit
        Ptranspose=((kron(pi_z',ones(N_a,N_a))).*(kron(exitprobs(1)*ones(N_z,1),Ptranspose)))+exitprobs(2)*((kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),(ones(N_a,1)*reshape(CondlProbOfSurvival,[1,N_a*N_z])).*Ptranspose)); % Add the appropriate for endogenous exit
    end
end

%% The rest is essentially the same regardless of which simoption.parallel is being used
%SteadyStateDistKron=ones(N_a*N_z,1)/(N_a*N_z); % This line was handy when checking/debugging. Have left it here.
if simoptions.parallel==2
    StationaryDistKronOld=zeros(N_a*N_z,1,'gpuArray');
else
    StationaryDistKronOld=zeros(N_a*N_z,1);
end
currdist=sum(abs(StationaryDistKron.pdf-StationaryDistKronOld));
counter=0;

% Switch into 'mass times pdf' form, and work with that until get
% convergence, then switch solution back into seperate mass and pdf form for output.
StationaryDistKron_pdf=StationaryDistKron.mass*StationaryDistKron.pdf; % Make it the pdf

while currdist>simoptions.tolerance && counter<simoptions.maxit

    for jj=1:100
       %% Following line is essentially the only change that entry and exit require to the actual iteration
        % Note that it works with cdf, rather than pdf. So there are also some lines just pre and post to do that.

        % Two steps of the Tan improvement [Has not yet been implemented for firm entry/exit]
        % StationaryDistKron_pdf=reshape(Gammatranspose*StationaryDistKron_pdf,[N_a,N_z]); %No point checking distance every single iteration. Do 100, then check.
        % StationaryDistKron_pdf=reshape(StationaryDistKron_pdf*pi_z_sparse,[N_a*N_z,1]);

        StationaryDistKron_pdf=MassOfNewAgents*DistOfNewAgentsKron+Ptranspose*StationaryDistKron_pdf; %No point checking distance every single iteration. Do 100, then check.
        % Note: Exit, captured in the CondlProbOfSurvival is already included into Ptranspose when it is created.
    end
    StationaryDistKronOld=StationaryDistKron_pdf;

    StationaryDistKron_pdf=MassOfNewAgents*DistOfNewAgentsKron+Ptranspose*StationaryDistKron_pdf; %No point checking distance every single iteration. Do 100, then check.
    
    currdist=sum(abs(StationaryDistKron_pdf-StationaryDistKronOld));
    % Note: I just look for convergence in the pdf and 'assume' the mass will also have converged by then. I should probably correct this.
    
    counter=counter+1;
    if simoptions.verbose==1
        if rem(counter,50)==0
            fprintf('StationaryDist_Case1: after %i iterations the current distance is %8.4f (tolerance=%8.4f) \n', counter, currdist, simoptions.tolerance)            
        end
    end
end
counter

% Turn it into the 'mass and pdf' format required for output.
StationaryDistKron.mass=sum(sum(StationaryDistKron.pdf));
StationaryDistKron.pdf=StationaryDistKron.pdf/StationaryDistKron.mass; % Make it the pdf


if simoptions.parallel>=3 % Solve with sparse matrix
    StationaryDistKron.pdf=full(StationaryDistKron.pdf);
    if simoptions.parallel==4 % Solve with sparse matrix, but return answer on gpu.
        StationaryDistKron.pdf=gpuArray(StationaryDistKron.pdf);
    end
end

if ~((100*counter)<simoptions.maxit)
    disp('WARNING: SteadyState_Case1 stopped due to reaching simoptions.maxit, this might be causing a problem')
end 

end
