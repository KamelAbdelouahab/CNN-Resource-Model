clc;
clear all;
close all;

J_macs = [];
J_nes  = [];
J_acts = [];
memory_wo_nefs = [];
memory_w_nefs = [];
numParams = [];
convLayerName = {};

%% Load caffe model
% bvlc_LeNet
% protoFile = '/home/kamel/Seafile/Kamel/leNet/caffe/lenet.prototxt';
% modelFile = '/home/kamel/Seafile/Kamel/leNet/caffe/lenet.caffemodel';
% AlexNet
protoFile = '/home/kamel/Seafile/Kamel/alexNet_compressed/caffe/alexNet.prototxt';
modelFile = '/home/kamel/Seafile/Kamel/alexNet_compressed/caffe/alexNet_compressed.caffemodel';
cnn       = caffe.Net(protoFile,modelFile,'test');


nBits = 8;
scaleFactor = 2 ^ (nBits-1) - 1;
layerNames = cnn.layer_names;

% Remove input, relu, dropout and fc from layer names
for i=1:numel(layerNames)
    if (strncmp('input',layerNames{i},4)  || ...
        strncmp('relu' ,layerNames{i},4)  || ...
        strncmp('norm' ,layerNames{i},4)  || ...
        strncmp('drop' ,layerNames{i},4)  || ...
        strncmp('fc'   ,layerNames{i},2) ...
        )
            layerNames{i} = [];
    end;
end;

layerNames = layerNames(~cellfun('isempty',layerNames));
layerNames = ['data'; layerNames]
for layerIndex=2:numel(layerNames)
    layerName   = layerNames{layerIndex};
    layer       = cnn.blobs(layerName).get_data();
    
    % Need to access last layer 
    prevLayerName = layerNames{layerIndex - 1};
    prevLayer     = cnn.blobs(prevLayerName).get_data();
    
    % test if layer is conv layer
    if (strncmp('conv',layerName,4));
        convLayerName = [convLayerName; layerName];
        w = cnn.params(layerName,1).get_data();
        [K K C N] = size(w);
        imageWidth = size(prevLayer,1);
        prevN = size(prevLayer,4);
%         if ((imageWidth-K-1) < K + 2)
%             tapsWidth = 0;
%         else
            tapsWidth = (imageWidth-K+1);
%         end;
    
        memory_wo_nef = nBits * N * prevN * (K-1)* tapsWidth;
        memory_w_nef  = nBits     * prevN * (K-1)* tapsWidth;
        memory_wo_nefs = [memory_wo_nefs;memory_wo_nef];
        memory_w_nefs = [memory_w_nefs;memory_w_nef];
    end;
end;
costMemory_value = [memory_wo_nefs memory_w_nefs];

figure()
bar(costMemory_value);
Labels = convLayerName;
set(gca, 'XTickLabel', Labels);
set(gca, 'YScale', 'log')
grid on;
ylabel('Required Memory (Bits)');
legend('w/ nef', 'wo/ nef','Location','Northeast');

