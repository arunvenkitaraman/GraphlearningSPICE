%% Estimation of graph for Temperature (Nov to Dec 2017) data using OLspice 
% Arun Venkitaraman  2017-03-22
% A. Venkitaraman and D. Zachariah, "Learning Sparse Graphs for Prediction of Multivariate Data Processes," 
%in IEEE Signal Processing Letters, vol. 26, no. 3, pp. 495-499, March 2019.
% https://ieeexplore.ieee.org/document/8629923

close all
clear all

p=45;
niter=10; % No fo SPICE iterations, typically 10 to 20 is fairly good.
Monte=10; % No of Monte carlo runs
ls_reg=1; % Regularization if needed for regularized LS 

load('city45data.mat'); % Temp signals
load('city45T.mat'); % Geodesic distances between cities

Xfull=T';
Xfull=Xfull-1*mean(Xfull,1); % Mean subtraction to zero

% Xfull is a matrix with graph signals arranged row-wise, ie, 
% dimensions: No of signals X no of nodes

A=A45; % Geodesic distance matrix in actual distances
A=A.^2; 
A=A/mean(A(:));
A=exp(-1*A);
A=A-diag(diag(A));
B0=A/max(abs(A(:)));


nsamp_vec=[10 20 30]; % No of training samples
nlen=length(nsamp_vec);

for s=1:Monte % For every Monte run
    for ns=1:nlen % For every training sample size
        n=nsamp_vec(ns);
        ttrain=randperm(62,n); % random training samples
        ttest=setdiff((1:62),ttrain);
        X=Xfull(ttrain,1:p);
        X=X-mean(X,1); % Needed?
        %
        Bhat=zeros(p);
        Bhat_ls=zeros(p);
        
        
        % tic
        for nod=1:p
            
            y=X(:,nod);
            H=X(:,[1:nod-1 nod+1:p]);
            
            [theta_hat, ~] = func_onlinespice(y,H,niter,1); %OLSPICE
            %theta_hat=zeros(p,1);
            theta_hat=(theta_hat');
            
            theta_hat_ls=pinv(H)*y;
            %theta_hat_ls=inv(H'*H+ls_reg*eye(p-1))*H'*y; % LS estimate may need regularization
            theta_hat_ls=(theta_hat_ls');
            
            Bhat(nod,:)=[theta_hat(1:nod-1) 0 theta_hat(nod:p-1)];
            Bhat_ls(nod,:)=[theta_hat_ls(1:nod-1) 0 theta_hat_ls(nod:p-1)];
            
        end
       
       
        %% Prediction results
        nodes_a=[1:5];
        nodes_b=[6:45];
        for ntest=1:20
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
NMSE=10*log10([mean(Mse1,2)./mean(En,2) mean(Mse2,2)./mean(En,2) mean(Mse3,2)./mean(En,2)])

toc