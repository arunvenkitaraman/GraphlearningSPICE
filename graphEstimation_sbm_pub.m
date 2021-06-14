%% Estimation of graph for EEG data using OLspice
% Arun Venkitaraman  2017-03-22
% A. Venkitaraman and D. Zachariah, "Learning Sparse Graphs for Prediction of Multivariate Data Processes," 
%in IEEE Signal Processing Letters, vol. 26, no. 3, pp. 495-499, March 2019.
% https://ieeexplore.ieee.org/document/8629923

close all
clear all

p=10;
niter=10; % No fo SPICE iterations, typically 10 to 20 is fairly good.
Monte=1; % No of Monte carlo runs
ls_reg=1; % Regularization if needed for regularized LS 


B0=[    0    0.2405    0.4750         0    0.0423         0         0         0         0         0;
    0.2405         0    0.3145         0    0.0942         0         0         0         0         0;
    0.4750    0.3145         0         0         0    0.5000         0         0         0         0;
         0         0         0         0    0.2453         0         0         0         0         0;
    0.0423    0.0942         0    0.2453         0         0         0         0         0         0;
         0         0         0         0         0         0    0.4150         0    0.0185    0.3947;
         0         0         0         0         0    0.4150         0         0    0.4077    0.4732;
         0         0         0         0         0         0         0         0    0.0149    0.2055;
         0         0         0         0         0    0.0185    0.4077    0.0149         0         0;
         0    0.5000         0         0         0    0.3947    0.4732    0.2055         0         0;
];



Lam=1*diag(rand(1,p)); % The diagonal variances of generating data
mu=zeros(1,p);

e=mvnrnd(mu,Lam,10000); % Samples generated
Xfull=(eye(p)-B0)\e';
Xfull=Xfull';


%Xfull=Xfull-1*mean(Xfull,1); % Mean subtraction to zero

nsamp_vec=[10 100 500]; % training sample sizes
nlen=length(nsamp_vec);


for s=1:Monte % For every Monte run
    for ns=1:nlen % For every training sample size
        n=nsamp_vec(ns);
        ttrain=randperm(9000,n); % random training samples
        ttest=setdiff((1:9000),ttrain);
        X=Xfull(ttrain,1:p);
        X=X-mean(X,1); %Needed?
        Bhat=zeros(p);
        Bhat_ls=zeros(p);
        
        % tic
        for nod=1:p
            
            y=X(:,nod);
            H=X(:,[1:nod-1 nod+1:p]);
            
            [theta_hat, ~] = func_onlinespice(y,H,niter,1);
            %theta_hat=zeros(p,1);
            theta_hat=(theta_hat');
            
            theta_hat_ls=pinv(H)*y;
            %theta_hat_ls=inv(H'*H+ls_reg*eye(p-1))*H'*y; % LS is bad without regularization
            theta_hat_ls=-(theta_hat_ls');
            
            Bhat(nod,:)=[theta_hat(1:nod-1) 0 theta_hat(nod:p-1)];
            Bhat_ls(nod,:)=[theta_hat_ls(1:nod-1) 0 theta_hat_ls(nod:p-1)];
            
        end
        
        
        %% Prediction results
        nodes_a=[1:2:p]; %
        nodes_b=setdiff(1:p,nodes_a); 
        
        
        for ntest=1:500
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
NMSE= 10*log10([mean(Mse1,2)./mean(En,2) mean(Mse2,2)./mean(En,2) mean(Mse3,2)./mean(En,2)])
toc