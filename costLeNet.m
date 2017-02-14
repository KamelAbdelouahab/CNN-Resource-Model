clc;
clear all;
close all;

nBits = 8;
scaleFactor = 2 ^ (nBits-1) - 1;

% Load LeNet
protoFile = '/home/kamel/Seafile/Kamel/leNet/caffe/lenet.prototxt';
modelFile = '/home/kamel/Seafile/Kamel/leNet/caffe/lenet.caffemodel';
cnn       = caffe.Net(protoFile,modelFile,'test');

% Static for now
w1 = cnn.params('conv1',1).get_data();
w2 = cnn.params('conv2',1).get_data();


% %% Compute Layer Hardware Cost 
% $J_{Layer} = C J_{ne} + N J_{act} + \sum \sum \sum J_{mult}(w)$ 

% Layer conv1:
[K K C N] = size(w2);

J_mac  = 0;
J_ne   = neCost(K);
J_act  = actCost(C); 

for n=1:N;
    for c=1:C;
        J_mac = J_mac + 47;
        for kx=1:K;
            for ky=1:K;
                  wRounded= round(scaleFactor*w2(kx,ky,c,n));
                  J_mac = J_mac + multCost(wRounded);
            end;
        end;
    end;
end;

% J_mac = J_mac + 0.3229*N*C*K*K;
J = C*J_ne + N*J_act + J_mac;

% Experimental values : w1
% J_neExp  =  128;
% J_macExp =  5195;
% J_actExp =  514;

% Experimental values : w2
J_neExp  =  1441;
J_macExp =  174485;
J_actExp =  7557;

err_ne  = 100* (C * J_ne - J_neExp)/J_neExp
err_mac = 100* (J_mac - J_macExp)/J_macExp
err_act = 100* (N * J_act - J_actExp)/J_actExp

err = 100*(J - J_neExp - J_macExp - J_actExp)  / (J_neExp + J_macExp + J_actExp) 



% J = C*J_ne + J_mac;
% kilos = 1;
% disp(['J_mac = ' num2str(round(J_mac/kilos))    ' kALM'])
% disp(['J_ne = '  num2str(round(C*J_ne/kilos))   ' kALM'])
% disp(['J_act = ' num2str(round(N*J_act/kilos))  ' kALM'])
% disp(['J = '     num2str(round(J/kilos))        ' kALM'])
% 
% labels = {'ne', 'act', 'mac'};
% pie([C*J_ne N*J_act J_mac],labels);
