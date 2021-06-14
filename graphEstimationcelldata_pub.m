%% Estimation of graph for cell signalling data using OLspice 
% Arun Venkitaraman  2017-03-22
% A. Venkitaraman and D. Zachariah, "Learning Sparse Graphs for Prediction of Multivariate Data Processes," 
%in IEEE Signal Processing Letters, vol. 26, no. 3, pp. 495-499, March 2019.
% https://ieeexplore.ieee.org/document/8629923

close all
clear all
tic

p=11; % Node size
niter=10; % No fo SPICE iterations, typically 10 to 20 is fairly good.
Monte=10; % No of Monte carlo runs
ls_reg=0; % Regularization if needed for regularized LS 

load('celldata1.mat');
load('celldata2.mat');
load('celldata3.mat');
load('celldata4.mat');
load('celldata5.mat');
load('celldata6.mat');
load('celldata7.mat');
load('celldata8.mat');
load('celldata9.mat');
load('cellA.mat');


Xfull=[celldata1;
    celldata2;
    celldata3;
    celldata4;
    celldata5;
    celldata6;
    celldata7;
    celldata8;
    celldata9
    ];


% Xfull is a matrix with graph signals arranged row-wise, ie, 
% dimensions: No of signals X no of nodes

%% OR
load('celldatafull.mat'); % All graph signals together as Xfull already
load('cellA.mat'); % Graph from Sachs et al.


Xfull=Xfull-1*mean(Xfull,1); % Mean subtracted to zero
nsamp_vec=[20 200 2000]; % No of training samples
nlen=length(nsamp_vec);

B0=cellA;

for s=1:Monte % For every Monte run
    for ns=1:nlen % For every training sample size
        n=nsamp_vec(ns);
        ttrain=randperm(7000,n);  % randomly generating training samples
        ttest=setdiff((1:7000),ttrain); % rest testing
        X=Xfull(ttrain,1:p);
        X=X-mean(X,1); % Needed?
        Bhat=zeros(p);
        Bhat_ls=zeros(p);
        
        
        
        for nod=1:p
            
            y=X(:,nod);
            H=X(:,[1:nod-1 nod+1:p]);
            
            
            [theta_hat, ~] = func_onlinespice(y,H,niter,1); %OLSPICE
            %theta_hat=zeros(p,1);
            theta_hat=(theta_hat');
            
            theta_hat_ls=pinv(H)*y;
            theta_hat_ls=inv(H'*H+ls_reg*eye(p-1))*H'*y;  % Regularization needed as simple LS really bad
            theta_hat_ls=(theta_hat_ls');
            
            Bhat(nod,:)=[theta_hat(1:nod-1) 0 theta_hat(nod:p-1)];
            Bhat_ls(nod,:)=[theta_hat_ls(1:nod-1) 0 theta_hat_ls(nod:p-1)];
        end
       
        
        %% Prediction results
        nodes_b=[3 8 9];
        nodes_a=setdiff(1:p,nodes_b);
        
        for ntest=1:3000
            x=Xfull(ttest(ntest),:)';
            x_b=x(nodes_b);
            x_a=x(nodes_a);
            
            xhat_a1=graphPrediction(nodes_a,nodes_b,B0,x_b);
            xhat_a2=graphPrediction(nodes_a,nodes_b,Bhat,x_b);
            xhat_a3=graphPrediction(nodes_a,nodes_b,Bhat_ls,x_b);
            mse1(ntest)=(norm(x_a-xhat_a1,2)^2);
            mse2(ntest)=(norm(x_a-xhat_a2,2)^2);
            mse3(ntest)=(norm(x_a-xhat_a3,2)^2);
            en(ntest)=norm(x_a,2)^2;
        end
        Mse1(ns,s)=(sum(mse1));
        Mse2(ns,s)=(sum(mse2));
        Mse3(ns,s)=sum(mse3);
        En(ns,s)=sum(en);
        
        
        
    end
end


% NMSE =['Truegraph' 'SPICE' 'LS']
NMSE=10*log10([mean(Mse1,2)./mean(En,2) mean(Mse2,2)./mean(En,2) mean(Mse3,2)./mean(En,2) ])

toc
