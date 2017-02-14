clc;
clear all;
close all;

J_macs = [];
J_nes  = [];
J_acts = [];
mem_wo_nefs = [];
mem_w_nefs = [];
numParams = [];

%% Load AlexNet caffe model
protoFile = '/home/kamel/Seafile/Kamel/alexNet_compressed/caffe/alexNet.prototxt';
% modelFile = '/home/kamel/Seafile/Kamel/alexNet/caffe/bvlc_alexnet.caffemodel';
modelFile = '/home/kamel/Seafile/Kamel/alexNet_compressed/caffe/alexNet_compressed.caffemodel';
cnn       = caffe.Net(protoFile,modelFile,'test');

%% Extract params
%  Static for now
layerNames = {'conv1','conv2','conv3','conv4','conv5'};
w1 = cnn.params('conv1',1).get_data();
w2 = cnn.params('conv2',1).get_data();
w3 = cnn.params('conv3',1).get_data();
w4 = cnn.params('conv4',1).get_data();
w5 = cnn.params('conv5',1).get_data();

%% Iterate over params
nBits = 8;
scaleFactor = 2 ^ (nBits-1) - 1;
sumMac = 47; % emperical result

W = {w1,w2,w3,w4,w5};
for layerIndex = 1:length(W)
    w = W{layerIndex};
    [K K C N] = size(w);
    
    %% Number of parameters
    numParam = numel(w);
    numParams = [numParams;numParam];
    
    %% Apply rounding
    w = round(scaleFactor*w);
    
    %% Compute hardware cost
    
    %% $J_{Layer} = C J_{ne} + N J_{act} + \sum \sum \sum J_{mult}(w)$ 
    J_mac   = 0;
    J_nes   = [J_nes;  C .* neCost(K)];      %Neigh extraction
    J_acts  = [J_acts; N .* actCost(C)];           %Activation
    for n=1:N;                                            %MAC
        for c=1:C;
            for kx=1:K;
                for ky=1:K;
                  J_mac = J_mac + multCost(w(kx,ky,c,n));
                end;
            end;
            J_mac = J_mac + sumMac;
        end;
    end;
    J_macs = [J_macs; J_mac];
end;


%% Save results
J = [J_nes J_macs J_acts];
save('alexNet_compressed.txt','J')

%% Display Results
figure()
% Display results by process
J_proc  = sum(J,1);
J_total = sum(J_proc);
labels = {'ne: ';'mac: ';'act: '};

subplot(1,2,1)
h = pie(J_proc);
hText = findobj(h,'Type','text'); % text object handles
percentValues = get(hText,'String'); % percent values
combinedtxt = strcat(labels,percentValues);
hText(1).String = combinedtxt(1);
hText(2).String = combinedtxt(2);
hText(3).String = combinedtxt(3);



% Display results by layer
J_layer = sum(J,2);
subplot(1,2,2)
labels = {'conv1: ';'conv2: ';'conv3: ';'conv4: ';'conv5: '};
h = pie(J_layer);
hText = findobj(h,'Type','text'); % text object handles
percentValues = get(hText,'String'); % percent values
combinedtxt = strcat(labels,percentValues);
hText(1).String = combinedtxt(1);
hText(2).String = combinedtxt(2);
hText(3).String = combinedtxt(3);
hText(4).String = combinedtxt(4);
hText(5).String = combinedtxt(5);


% Display with bars

figure()
bar(J,'stacked')
Labels = layerNames;
set(gca, 'XTickLabel', Labels);
ylabel('Logic Elements (ALM)')
% title('')
legend('NE','MAC','ACT','Location','Northwest')
dim = [0.15 0.4 0.2 0.3];
str = {'Model: AlexNet\_compressed', strcat('Total kALMs: ',num2str(round(J_total/1000)))};
annotation('textbox',dim,'String',str,'FitBoxToText','on','FontSize',8);