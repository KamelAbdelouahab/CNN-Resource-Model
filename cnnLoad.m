clc;
clear all;
close all;

compLoads = [];
memLoads = [];

convCompLoad = 0;
fcCompLoad = 0;

convMemLoad = 0;
fcMemLoad = 0;

%% Set fixed point precision
nBits = 8
scaleFactor = 2 ^ (nBits-1) - 1;


%% Load caffe model
% %AlexNet bvlc
% modelName = ' AlexNet bvlc';
% protoFile = '/home/kamel/Seafile/Kamel/alexNet_full/deploy.prototxt';
% modelFile = '/home/kamel/Seafile/Kamel/alexNet_full/bvlc_alexnet.caffemodel';

% % %Lenet 5
modelName = 'leNe5'
protoFile = '/home/kamel/Seafile/Kamel/leNet/caffe/lenet.prototxt';
modelFile = '/home/kamel/Seafile/Kamel/leNet/caffe/lenet.caffemodel';


cnn       = caffe.Net(protoFile,modelFile,'test');
layerNames = cnn.layer_names;

for i=1:numel(layerNames)
    if (strncmp('input',layerNames{i},4)  || ...
        strncmp('relu' ,layerNames{i},4)  || ...
        strncmp('norm' ,layerNames{i},4)  || ...
        strncmp('drop' ,layerNames{i},4)     ...
        )
            layerNames{i} = [];
    end;
end;

%% Extract params
layerNames = layerNames(~cellfun('isempty',layerNames));
% layerNames = ['data'; layerNames];


for layerIndex = 2:length(layerNames)
    
     % Need to access layer    
    layerName   = layerNames{layerIndex};
    layer       = cnn.blobs(layerName).get_data();
    
    % Need to access previous layer 
    prevLayerName = layerNames{layerIndex - 1};
    prevLayer     = cnn.blobs(prevLayerName).get_data();
    
    if (strncmp('conv',layerName,4))
        % Layer is conv layer
        [K K C N] = size (cnn.params(layerName,1).get_data());
        H         = size(prevLayer,1);
        convCompLoad =  convCompLoad + K*K*C*N*H*H + C*N*H*H;
%         compLoads = [compLoads; K*K*C*N*H*H + C*N*H*H];
        compLoads = [compLoads; K*K*C*N + C*N];
        
        convMemLoad = convMemLoad + nBits*K*K*C*N;
        memLoads = [memLoads; nBits*K*K*C*N];
    end;
    
%        if (strncmp('fc',layerName,2))
%         % Layer is fc layer
%         [C N] = size (cnn.params(layerName,1).get_data());
%         fcCompLoad = fcCompLoad + C*N + N;
%         compLoads = [compLoads; C*N + N;];
%  
%         fcMemLoad = fcMemLoad + nBits*C*N;
%         memLoads = [memLoads; nBits*C*N;];
%     end; 
end;

for i=1:numel(layerNames)
    if (strncmp('pool',layerNames{i},4)  || ...
        strncmp('prob' ,layerNames{i},4) || ...
        strncmp('data' ,layerNames{i},4)     ...
        )
            layerNames{i} = [];
    end;
end;
layerNames = layerNames(~cellfun('isempty',layerNames));

figure()
subplot(1,3,1)
pie(compLoads)
subplot(1,3,2)
pie(memLoads)
legend(layerNames)
% subplot(1,3,3)
axis off;

% subplot(1,2,1)
% pie([convMemLoad fcMemLoad]);
% 
% subplot(1,2,2)
% pie([convCompLoad fcCompLoad]);
% legend('convLayer','fcLayer');
   