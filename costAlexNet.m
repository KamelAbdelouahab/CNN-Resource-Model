clc;
clear all;
close all;

nBits = 8;
scaleFactor = 2 ^ (nBits-1) - 1;

% Load AlexNet
protoFile = '/home/kamel/Seafile/Kamel/alexNet_compressed/caffe/alexNet.prototxt';
modelFile = '/home/kamel/Seafile/Kamel/alexNet_compressed/caffe/alexNet_compressed.caffemodel';
% modelFile = '/home/kamel/Seafile/Kamel/alexNet_compressed/caffe/alexNet.caffemodel';

cnn       = caffe.Net(protoFile,modelFile,'test');

% Static for now
w1 = cnn.params('conv1',1).get_data();
% w2 = cnn.params('conv2',1).get_data();
% w3 = cnn.params('conv3',1).get_data();
% w4 = cnn.params('conv4',1).get_data();
% w5 = cnn.params('conv5',1).get_data();

% histogram(w1,200)
% %% Compute Layer Hardware Cost 
% $J_{Layer} = C J_{ne} + N J_{act} + \sum \sum \sum J_{mult}(w)$ 

% Layer conv1:
w = w1;

[K K C N] = size(w);
% 

J_mac  = 0;
J_ne   = neCost(K);
J_act  = actCost(C); 
% 
for n=1:N;
    for c=1:C;
        
        zk = 0; % null weight
        for kx=1:K;
            for ky=1:K;
            wRounded= round(scaleFactor*w(kx,ky,c,n));
                  if (wRounded==0)
                      s = 0;
%                       zk = zk+1;
%                   else
%                       s = sumCost(K^2 - zk);
                  end;
                  J_mac = J_mac + multCost(wRounded);
            end;
        end;
        J_mac = J_mac + 47;
    
    end;
end;
% J_mac/1000
% 
J = C*J_ne + N*J_act + J_mac;
% J = C*J_ne + J_mac;
disp(['J_mac = ' num2str(round(J_mac/1000))    ' kALM'])
disp(['J_ne = '  num2str(round(C*J_ne/1000))   ' kALM'])
disp(['J_act = ' num2str(round(N*J_act/1000))  ' kALM'])
disp(['J = '     num2str(round(J/1000))        ' kALM'])
% 
% labels = {'ne', 'act', 'mac'};
% pie([C*J_ne N*J_act J_mac],labels);

% Experimental values
J_neExp  =  1654;
J_macExp =  162748;
J_actExp =  4504;


err_ne  = 100* (C * J_ne - J_neExp)/J_neExp
err_mac = 100* (J_mac - J_macExp)/J_macExp
err_act = 100* (N * J_act - J_actExp)/J_actExp

err = 100*(J - J_neExp - J_macExp - J_actExp)  / (J_neExp + J_macExp + J_actExp) 

