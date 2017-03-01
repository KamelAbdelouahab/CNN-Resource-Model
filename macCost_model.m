clc;
clear all;
close all;

J_macs = [];
J_nes  = [];
J_acts = [];
mem_wo_nefs = [];
mem_w_nefs = [];
numParams = [];


% Set fixed point precision
nBits = 5
scaleFactor = 2 ^ (nBits-1) - 1;
thresh = 80; %% Take 


% Load caffe model

% %AlexNet compressed
% Put the path to your proto and caffemodel and quartus fir report here
modelName   = 'AlexNet compressed'
protoFile   = '/home/kamel/Seafile/Kamel/alexNet_compressed/caffe/alexNet.prototxt';
modelFile   = '/home/kamel/Seafile/Kamel/alexNet_compressed/caffe/alexNet_compressed.caffemodel';
fitFilename = '/home/kamel/Seafile/Kamel/Reports/alexnet_5bits.rpt';

% %Lenet 5
% modelName = 'leNe5'
% protoFile = '/home/kamel/Seafile/Kamel/leNet/caffe/lenet.prototxt';
% modelFile = '/home/kamel/Seafile/Kamel/leNet/caffe/lenet.caffemodel';

cnn       = caffe.Net(protoFile,modelFile,'test');

%  Extract params of first layer
w = cnn.params('conv1',1).get_data();
[K K C N] = size(w);

%  Apply rounding
w = round(scaleFactor*w);

% Reshape to be same as HW model
w_rep = reshape(w,[K,K,C*N]);
w_sum = zeros(C*N,1);
w_mul = zeros(C*N,1);

% number of Null kernel values is prop to J_sum 
% number of Null AND POWEROF2 kernel values is prop to J_mul

for i=1:C*N
    [w_sum(i) w_mul(i)] = alm_metric(w_rep(:,:,i));
end;




J = getALM2(fitFilename);
J = sortrows(J);

J_sum = J(:,3);
J_mac = J(:,2);
J_mul = J(:,2) - J(:,3);

% number of Null kernel values is prop to J_sum 
% number of Null AND POWEROF2 kernel values is prop to J_mul


% Number of macs too
% scatter(w_mul,J_mac);

% Linear models
lm_sum = fitlm(w_sum,J_sum)
coef_sum = lm_sum.Coefficients.Estimate;
lm_mul = fitlm(w_mul,J_mul)
coef_mul = lm_mul.Coefficients.Estimate;
x = [20:90]';
x = [ones(length(x),1) x];


subplot(2,1,1)
plot(x(:,2),x*coef_sum,'r');
grid on;
hold on;
scatter(w_sum,J_sum);
axis ([0 100 0 250])
ylabel('ALM usage');
xlabel(' % of null values in kernel');
title('Accumulation HW cost');
subplot(2,1,2)
plot(x(:,2),x*coef_mul,'r');
grid on;
hold on;
scatter(w_mul,J_mul);
axis ([0 100 0 250])
ylabel('ALM usage');
xlabel(' % of null and pow2 values in kernel');
title('Multiplciation HW cost');

% scatter(w_n,J_sum);
% grid on;
% hold on;

% x = w_n;
% y= J_sum;
% scatter(x,y);
% expEqn = 'a*(1-exp(b*x))';
% startPoints = [600 0.1];
% exclude = x>90;

% m = fit(x,y,expEqn,'Start',startPoints,'Exclude', exclude)
% m = fit(x,y,'poly2');
% % lm = fitlm(x,y)
% hold on;plot(m,x,y);

% Do a threshold
% J = J(w_n<thresh,:);
% w_n = w_n(w_n<thresh);
% Plot
% scatter(w_n,J(:,4));
% hold on;
% grid on;
% scatter(w_n,J(:,3));
% legend('Multiplication', 'Accumulation');
% sum11 = mean(J(:,3));
% 
% % %% Linar model
% % % % X = [ones(length(w_n),1) w_n sqrt(w_n)];
% X = [ones(length(w_n),1) w_n];
% % X = w_n;
% yMul = J(:,4);
% yAdd = J(:,3);
% mMul = X\yMul;
% mAdd = X\yAdd;
% plot(w_n,X*mMul,'b');
% plot(w_n,X*mAdd,'r');
% legend('Multiplication', 'Accumulation');

% figure()
% y = J(:,2);
% X = [ones(length(x),1) x];
% M = X\y;
% scatter(x,y);
% % % scatter(w_n,J(:,2)- J(:,3));
% % 
% hold on
% plot(x,X*M,'r');
% xlabel(' % Weights with a null / power-of-two value');
% ylabel(' Hardware resource of a MAC operation (ALM)');
% 
% 
% % x= X(:,2);
% % lm = fitlm(x,y)
% 
% % % Extract params from second layer
% w = cnn.params('conv2',1).get_data();
% [K K C N] = size(w);
% % 
% % Apply rounding
% w = round(scaleFactor*w);
% % 
% % % Reshape to be same as HW model
% w_rep = reshape(w,[K,K,C*N]);
% w_n = zeros(C*N,1);
% % 
% for i=1:C*N
%     w_n(i) = alm_metric(w_rep(:,:,i));
% end;
% % 
% X2 = [ones(length(w_n),1) w_n];
% 
% % % Update model with 3x3 parallel adder cost instead of 11x11
% M(1) = M(1) .* 9./ 121; 
% % % X2 = w_n;
% y2 = X2*M;
% % plot(X2,y2,'g');
% J_mac2 = sum(y2)
%     
