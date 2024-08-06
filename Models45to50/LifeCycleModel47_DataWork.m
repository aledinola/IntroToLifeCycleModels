%% Empirical data work for Life-Cycle Model 47
% You first need to download data from PSID (explained in third section of
% codes, starting line 41), then this will import the data and run the whole empirical analysis.

% Note: Our model has people working ages 20 to 64, retired 65+.
% So we will get earnings data on ages 20 to 64.

% When calculating earnings profiles from the data we need to adjust for
% inflation (so real earnings), and we also need to either use fixed
% effects for cohorts, or fixed effects for time (year). We use cohort fixed
% effects as our main estimates of age-conditional mean earnings 
% (and we estimate the covariance matrix of these), and just to show it we
% also do the time fixed effects (but skip the covariance matrix for these
% as we will not use them in our estimation).

% For an explanation of the issue of cohort-time-age and how we can only
% control for two of these (we use cohort and age, and we demo time and
% age) see
% AAAAAAAAAAAAAAAAAAAAAAAAAAA

%% Description of the data used
% Uses PSID data on earnings, 1969-1992 family files
% Earnings of male household heads
% Ages 20 to 64 (inclusive)
% Restrict sample to those with strictly positive earnings
% Exclude the survey of economic opportunities (SEO) sample which is a
% subsample of the PSID that oversamples the poor.
%
% Broad sample: all males who are currently working, temporarily laid off,
% looking for work but currently unemployed, students, but does not include
% retirees.
% Narrow sample: equals the broad sample, less those unemployed or
% temporarily laid off
% [The estimation uses the results for the broad sample.]
%
% Given all the above sample selection criteria, the average and standard
% deviation of the number of observations per panel-year are 2137 and 131,
% respectively.


%% Download the data from PSID, then import
% Go to PSID and look for previous "carts"
% https://simba.isr.umich.edu/DC/c.aspx
% Search for and download the cart called 
% "325247 - HVY2006" associated with the email address "robertdkirkby@gmail.com"
% This will download a zip file, unzip it and put the files in the same
% folder as this current m-file.
% Rename the file which is .sps to _sps.txt
% Rename the file which is _format.sps to _formate_sps.txt
% (if you don't rename these ImportPSIDdata will just give a warning telling you to do so)
% Then the following line of code will import that PSID data
MyPSIDdataset=ImportPSIDdata('J325247');

% Reading the PSID documentation we learn that:
% Whether or not family is in SEO is based on data variable "Family Number"
% Let's do a fully described example of creating a binary variable indicating SEO

% MyPSIDdataset is a structure containing both the data and all the indentifying info 
% We look at it
MyPSIDdataset % shows variables (by code)
% We happen to already have learnt that V3 is the family number
MyPSIDdataset.V3
% We can see there it is the 1968 Family Number, the 'etext' explains what
% the interpretation is, and 'value' contains all the actual data
% In the 'etext' we read that values <3000 are the main sample, and higher
% than this is the SEO sample.
% The original MyPSIDdataset.V3.value are strings
% There is also an autogenerated attempt to turn these into numbers
% in 'value2' and we will use this
% We cannot just use MyPSIDdataset.V3.value2<=3000 because of how the
% values are set up (it only lists family number for some, not all)
% Instead I found family number 5001 is in column 9978
% And last main sample family seems to be 2930 in column 9536
% Double-checked there is not family 2931
% So I end up using
MainSampleIndicator=[ones(1,9600),zeros(1,length(MyPSIDdataset.V3.value2)-9600)]; % Note: has to be <=3000, as if I tried to create SEOIndicator using >3000 it would not handle NaN in the way I want

% Now we just grab the rest of the relevant data as a panel
% We want earnings, looking through the first few variables we see that V74
% has the etext: {'(Labor part of farm income and business income, wages, bonuses, 
% overtime, commissions, professional practice, labor part of income from roomers and 
% boarders or business income (See editing instructions).)'}
% So we want this variable in this and other years. 
% Easiest way to figure out the other years is going to the PSID website
% data search for variables
% https://simba.isr.umich.edu/DC/s.aspx
% We can search for V74 (make sure to tick the boxes below for PSID family-level and 1968)
% We can click the "i" info button next to V74 (in search results) and
% there is a field in what opens up that says
% "Years Available: 	[68]V74 [69]V514 [70]V1196 [71]V1897 [72]V2498 [73]V3051 [74]V3463 [75]V3863 
% [76]V5031 [77]V5627 [78]V6174 [79]V6767 [80]V7413 [81]V8066 [82]V8690 [83]V9376 [84]V11023 [85]V12372 
% [86]V13624 [87]V14671 [88]V16145 [89]V17534 [90]V18878 [91]V20178 [92]V21484 ..."
% So now we know all the variable names for earnings in each year [all of which were added to the cart we downloaded]
Earnings=[MyPSIDdataset.V74.value2; MyPSIDdataset.V514.value2; MyPSIDdataset.V1196.value2; MyPSIDdataset.V1897.value2; MyPSIDdataset.V2498.value2;...
    MyPSIDdataset.V3051.value2; MyPSIDdataset.V3463.value2; MyPSIDdataset.V3863.value2; MyPSIDdataset.V5031.value2; MyPSIDdataset.V5627.value2;...
    MyPSIDdataset.V6174.value2; MyPSIDdataset.V6767.value2; MyPSIDdataset.V7413.value2; MyPSIDdataset.V8066.value2; MyPSIDdataset.V8690.value2;...
    MyPSIDdataset.V9376.value2; MyPSIDdataset.V11023.value2; MyPSIDdataset.V12372.value2; MyPSIDdataset.V13624.value2; MyPSIDdataset.V14671.value2;...
    MyPSIDdataset.V16145.value2; MyPSIDdataset.V17534.value2; MyPSIDdataset.V18878.value2; MyPSIDdataset.V20178.value2; MyPSIDdataset.V21484.value2;...
    ];

% Now we also need the "employment status" (which is telling us about "who are currently working, 
% temporarily laid off, looking for work but currently unemployed, students, but does not include retirees.")
% Years Available: 	[68]V196 [69]V639 [70]V1278 [71]V1983 [72]V2581 [73]V3114 [74]V3528 [75]V3967 
% [76]V4458 [77]V5373 [78]V5872 [79]V6492 [80]V7095 [81]V7706 [82]V8374 [83]V9005 [84]V10453 [85]V11637 
% [86]V13046 [87]V14146 [88]V15154 [89]V16655 [90]V18093 [91]V19393 [92]V20693 ..."
EmploymentStatus=[MyPSIDdataset.V196.value2; MyPSIDdataset.V639.value2; MyPSIDdataset.V1278.value2; MyPSIDdataset.V1983.value2; MyPSIDdataset.V2581.value2;...
    MyPSIDdataset.V3114.value2; MyPSIDdataset.V3528.value2; MyPSIDdataset.V3967.value2; MyPSIDdataset.V4458.value2; MyPSIDdataset.V5373.value2;...
    MyPSIDdataset.V5872.value2; MyPSIDdataset.V6492.value2; MyPSIDdataset.V7095.value2; MyPSIDdataset.V7706.value2; MyPSIDdataset.V8374.value2;...
    MyPSIDdataset.V9005.value2; MyPSIDdataset.V10453.value2; MyPSIDdataset.V11637.value2; MyPSIDdataset.V13046.value2; MyPSIDdataset.V14146.value2;...
    MyPSIDdataset.V15154.value2; MyPSIDdataset.V16655.value2; MyPSIDdataset.V18093.value2; MyPSIDdataset.V19393.value2; MyPSIDdataset.V20693.value2;...
    ];

% And we want the age of the household head, V117 is one of them, and from PSID website we get
% "Years Available: 	[68]V117 [69]V1008 [70]V1239 [71]V1942 [72]V2542 [73]V3095 [74]V3508 [75]V3921 
% [76]V4436 [77]V5350 [78]V5850 [79]V6462 [80]V7067 [81]V7658 [82]V8352 [83]V8961 [84]V10419 [85]V11606 
% [86]V13011 [87]V14114 [88]V15130 [89]V16631 [90]V18049 [91]V19349 [92]V20651 ..."
AgeOfHead=[MyPSIDdataset.V117.value2; MyPSIDdataset.V1008.value2; MyPSIDdataset.V1239.value2; MyPSIDdataset.V1942.value2; MyPSIDdataset.V2542.value2;...
    MyPSIDdataset.V3095.value2; MyPSIDdataset.V3508.value2; MyPSIDdataset.V3921.value2; MyPSIDdataset.V4436.value2; MyPSIDdataset.V5350.value2;...
    MyPSIDdataset.V5850.value2; MyPSIDdataset.V6462.value2; MyPSIDdataset.V7067.value2; MyPSIDdataset.V7658.value2; MyPSIDdataset.V8352.value2;...
    MyPSIDdataset.V8961.value2; MyPSIDdataset.V10419.value2; MyPSIDdataset.V11606.value2; MyPSIDdataset.V13011.value2; MyPSIDdataset.V14114.value2;...
    MyPSIDdataset.V15130.value2; MyPSIDdataset.V16631.value2; MyPSIDdataset.V18049.value2; MyPSIDdataset.V19349.value2; MyPSIDdataset.V20651.value2;...
    ];

% Gender of household head (so can restrict sample to males), V119 is one of them, and from PSID website we get
% Years Available: 	[68]V119 [69]V1010 [70]V1240 [71]V1943 [72]V2543 [73]V3096 [74]V3509 [75]V3922 
% [76]V4437 [77]V5351 [78]V5851 [79]V6463 [80]V7068 [81]V7659 [82]V8353 [83]V8962 [84]V10420 [85]V11607 
% [86]V13012 [87]V14115 [88]V15131 [89]V16632 [90]V18050 [91]V19350 [92]V20652 ..."
GenderOfHead=[MyPSIDdataset.V119.value2; MyPSIDdataset.V1010.value2; MyPSIDdataset.V1240.value2; MyPSIDdataset.V1943.value2; MyPSIDdataset.V2543.value2;...
    MyPSIDdataset.V3096.value2; MyPSIDdataset.V3509.value2; MyPSIDdataset.V3922.value2; MyPSIDdataset.V4437.value2; MyPSIDdataset.V5351.value2;...
    MyPSIDdataset.V5851.value2; MyPSIDdataset.V6463.value2; MyPSIDdataset.V7068.value2; MyPSIDdataset.V7659.value2; MyPSIDdataset.V8353.value2;...
    MyPSIDdataset.V8962.value2; MyPSIDdataset.V10420.value2; MyPSIDdataset.V11607.value2; MyPSIDdataset.V13012.value2; MyPSIDdataset.V14115.value2;...
    MyPSIDdataset.V15131.value2; MyPSIDdataset.V16632.value2; MyPSIDdataset.V18050.value2; MyPSIDdataset.V19350.value2; MyPSIDdataset.V20652.value2;...
    ];

sizeonevariable=size(MyPSIDdataset.V119.value2);
Year=(1968:1:1992)'*ones(sizeonevariable);

% Because of how I restrict the sample I also want an indicator for the individual
IndividualIndicator=ones(25,1)*(1:1:length(MyPSIDdataset.V3.value2));


%% Create the sample (impose the sample restrictions)
% We have earnings
% First, an indicator for ages 20 to 64
I1=(18<=AgeOfHead).*(AgeOfHead<=64);
% Second, an indicator of male
I2=(GenderOfHead==1); % 1 is male, 2 is female, 9 is NA
% Third, an indicator for "currently working, temporarily laid off,
% looking for work but currently unemployed, students, but does not include retirees"
I3A=(EmploymentStatus==1)+(EmploymentStatus==2)+(EmploymentStatus==5); % 1=Working now, or laid off only temporarily, 2=Unemployed, 3=Retired, permanently disabled, 4=Housewife, 5=Student, 6=Other
% Or in the narrow sample, "equals the broad sample, less those unemployed or temporarily laid off"
I3B=(EmploymentStatus==1)+(EmploymentStatus==5);
% Fourth, exclude the SEO sample
I4=MainSampleIndicator;
% Fifth, those with strictly positive earnings
I5=(Earnings>0);
% Sixth, While I download all the 1968 data, we are only using 1969-1992. So impose that restriction
I6=(Year>=1969).*(Year<=1992);


% So indicators for the broad and narrow restricted samples are
I_broad=logical(I1.*I2.*I3A.*I4.*I5.*I6);
I_narrow=logical(I1.*I2.*I3B.*I4.*I5.*I6);

% Broad sample
BroadSample.Earnings=Earnings(I_broad);
BroadSample.Age=AgeOfHead(I_broad);
BroadSample.Year=Year(I_broad);
BroadSample.Individual=IndividualIndicator(I_broad);

% Narrow sample
NarrowSample.Earnings=Earnings(I_narrow);
NarrowSample.Age=AgeOfHead(I_narrow);
NarrowSample.Year=Year(I_narrow);
NarrowSample.Individual=IndividualIndicator(I_narrow);

% Get get an idea how restrictive each of these is
sum(sum(I1))
sum(sum(I2))
sum(sum(I3A))
sum(sum(I4)).*25 % Note: misleading to just sum(sum(I4)) as it only has household dimension, missing the year dimension
sum(sum(I5))
sum(sum(I_broad))

% Look again
sum(sum(I1.*I4))
sum(sum(I2.*I4))
sum(sum(I3A.*I4))
sum(sum(I5.*I4))
sum(sum(I_broad))
% Looks like age of head is the most restrictive thing here
% Look at marginal contributions of other others after we impose main sample and age 
sum(sum(I1.*I4))
sum(sum(I2.*I4.*I1))
sum(sum(I3A.*I4.*I1))
sum(sum(I5.*I4.*I1))
sum(sum(I_broad))
% Yeah, almost all of resticting the sample comes from main sample (not SEO) and age of head.


%% Take a look at some basics about our data set
% the "average and standard deviation of the number of observations per panel-year are 2028 and 189"

% There are 25 years, so the average number of observations per panel-year is
fprintf('Average number of observations per panel-year is %5.0f \n', numel(BroadSample.Earnings)/25)

[cnt_unique, ~] = hist(BroadSample.Year,unique(BroadSample.Year));
fprintf('The standard deviation of the number of observations per panel year is %5.0f \n', std(cnt_unique))


%% Convert to real earnings, and organize them by age and year

% First, we need to convert nominal earnings into real earnings.
% Do this using CPI
CPI=getFredData('CPIAUCSL','1968-01-01','1992-12-31','lin','a','avg');
% I will use 1969-1992 data, and renormalize CPI so that 1969=1
CPIdata=CPI.Data(2:end)./CPI.Data(2);
% Now use this to get real earnings
BroadSample.RealEarnings=zeros(size(BroadSample.Earnings));
for aa=1:length(BroadSample.Earnings)
    BroadSample.RealEarnings(aa)=BroadSample.Earnings(aa)/CPIdata(BroadSample.Year(aa)-1968);
end

% Get the earnings by age and year
MaxObs=1000; % Just a guess, but each year has only 2000-ish observations, so seems like this should be way more than in a given age for a given year
RealEarnings_jjtt=nan(MaxObs,64-19,1992-1968); % Note: NaN
Counter_jjtt=zeros(64-19,1992-1968);
for aa=1:length(BroadSample.RealEarnings)
    tt=BroadSample.Year(aa)-1968;
    for jj=20:64
        if BroadSample.Age(aa)>=(jj-2) && BroadSample.Age(aa)<=(jj+2) % age is appropriate
            Counter_jjtt(jj-19,tt)=Counter_jjtt(jj-19,tt)+1;
            RealEarnings_jjtt(Counter_jjtt(jj-19,tt),jj-19,tt)=BroadSample.RealEarnings(aa);
        end
    end
end

% Earnings are currently in (real 1969) dollars 
% Want to run the model in thousands of dollars, not in dollars, so switch
% to measuring earnings in thousands of dollars
RealEarnings_jjtt=RealEarnings_jjtt/1000;


%% Construction of age-profiles of mean earnings

% Regress log earnings on age and cohort effects
J=64-19;
T=1992-1968;
AgeFixedEffects=zeros(J,T,J); % J variables (each column) which each have J*T observations (the rows)
for jj=1:J
    AgeFixedEffects(jj,:,jj)=1;
end
AgeFixedEffects=reshape(AgeFixedEffects,[J*T,J]);
Ncohorts=J+(T-1); % cohort is effectively just age minus year
CohortFixedEffects=zeros(J,T,Ncohorts);
for jj=1:J
    for tt=1:T
        CohortFixedEffects(jj,tt,jj-tt+T)=1; % cohort is just age minus year [the T is youngest jj in last tt gives index of 1]
    end
end
CohortFixedEffects=reshape(CohortFixedEffects,[J*T,Ncohorts]);

% Cohort-fixed-effect regression
ydata=reshape(RealEarnings_jjtt,[MaxObs*(64-19)*(1992-1968),1]);
Xdata=[repelem(AgeFixedEffects,MaxObs,1),repelem(CohortFixedEffects,MaxObs,1)];
b = regress(ydata,Xdata);
B=b(1:J); % b is (J+Ncohorts)-by-1

% Finally, we plot the fitted age effects
MeanEarningsProfile_CFE=B; % Note: because the AgeFixedEffects are just values of 1, we can skip doing "B*1" for each age.

% Plot the fitted mean earnings
figure(1);
plot(20:1:64,B)
title('Fitted mean earnings profile')

% Unfortunately the ages 62+ don't look good. Is likely a selection bias issue 
% but I think I will just drop them from the targets as that is a bit easier.

%% Calculate the covariance matrix of the data moments
% We can take advantage of the fact that the covariance matrix is
% essentially just earnings squared with zeros on the non-diagonals.

% Cohort-fixed-effect regression
ydata=reshape(RealEarnings_jjtt.^2,[MaxObs*(64-19)*(1992-1968),1]); % square of earnings
Xdata=[repelem(AgeFixedEffects,MaxObs,1),repelem(CohortFixedEffects,MaxObs,1)];
b2 = regress(ydata,Xdata);
B2=b2(1:J); % b2 is (J+Ncohorts)-by-1

% Finally, we plot the fitted age effects
MeanEarningsSquare=B2; % Note: because the AgeFixedEffects are just values of 1, we can skip doing "B*1" for each age.

CovarMatrixDataMoments_CFE=diag(MeanEarningsSquare);


%% Alternative views of age effects (not used in estimation)
% Redo, but instead of cohort fixed effects, we will instead use time fixed effects.
% Then plot the mean earnings profile, for both cohort fixed effects and
% time fixed effects, so we can see the difference.

TimeFixedEffects=zeros(J,T,T-1); % T-1 variables (each column) which each have J*T observations (the rows)
for tt=1:T-1
    TimeFixedEffects(:,tt,tt)=1;
end
TimeFixedEffects=reshape(TimeFixedEffects,[J*T,T-1]);
% Have to use only T-1 variables here

% Time-fixed-effect regression
ydata=reshape(RealEarnings_jjtt,[MaxObs*(64-19)*(1992-1968),1]);
Xdata=[repelem(AgeFixedEffects,MaxObs,1),repelem(TimeFixedEffects,MaxObs,1)];
b3 = regress(ydata,Xdata);
B3=b3(1:J); % b3 is (J+T)-by-1

MeanEarningsProfile_TFE=B3;
% TFE=time fixed effect

figure(2);
plot(20:1:64,MeanEarningsProfile_CFE,20:1:64,MeanEarningsProfile_TFE)
title('Mean Earnings')
xlabel('Age')
legend('Cohort FE','Time FE','location','southeast')

%% Calculate the covariance matrix of the data moments, for time-effects
% We can take advantage of the fact that the covariance matrix is
% essentially just earnings squared with zeros on the non-diagonals.

% Time-fixed-effect regression
ydata=reshape(RealEarnings_jjtt.^2,[MaxObs*(64-19)*(1992-1968),1]); % square of earnings
Xdata=[repelem(AgeFixedEffects,MaxObs,1),repelem(TimeFixedEffects,MaxObs,1)];
b2 = regress(ydata,Xdata);
B4=b2(1:J); % b2 is (J+Ncohorts)-by-1

% Finally, we plot the fitted age effects
MeanEarningsSquare_TFE=B4; % Note: because the AgeFixedEffects are just values of 1, we can skip doing "B*1" for each age.

CovarMatrixDataMoments_TFE=diag(MeanEarningsSquare_TFE);


%% Done. Time for the model!

% Note: the things we want for model are either the cohort-fixed-effects
% estimates for mean-earnings life-cycle profile, and the covar matrix.
% MeanEarningsProfile_CFE
% CovarMatrixDataMoments_CFE
% Or the time-fixed effects estimates
% MeanEarningsProfile_TFE
% CovarMatrixDataMoments_TFE

