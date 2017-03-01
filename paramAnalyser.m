function Y = paramAnalyser(modelName,protoFile,modelFile);
nBits = 8;
scaleFactor = 2 ^ (nBits-1) - 1;
W = [];

% %% Load caffe model
% 
% %AlexNet compressed
modelName = 'AlexNet compressed'
protoFile = '/home/kamel/Seafile/Kamel/alexNet_compressed/caffe/alexNet.prototxt';
modelFile = '/home/kamel/Seafile/Kamel/alexNet_compressed/caffe/alexNet_compressed.caffemodel';
% 
% % %AlexNet bvlc
% % modelName = ' AlexNet bvlc';
% % protoFile = '/home/kamel/Seafile/Kamel/alexNet_compressed/caffe/alexNet.prototxt';
% % modelFile = '/home/kamel/Seafile/Kamel/alexNet/caffe/bvlc_alexnet.caffemodel';
% 
% % % %Lenet 5
% % modelName = 'leNe5'
% % protoFile = '/home/kamel/Seafile/Kamel/leNet/caffe/lenet.prototxt';
% % modelFile = '/home/kamel/Seafile/Kamel/leNet/caffe/lenet.caffemodel';

cnn       = caffe.Net(protoFile,modelFile,'test');
layerNames = cnn.layer_names;

%% Remove input, relu, dropout and fc from layer names

for i=1:numel(layerNames)
    if (strncmp('input',layerNames{i},4)  || ...
        strncmp('relu' ,layerNames{i},4)  || ...
        strncmp('data' ,layerNames{i},4)  || ...
        strncmp('norm' ,layerNames{i},4)  || ...
        strncmp('drop' ,layerNames{i},4)  || ...
        strncmp('pool' ,layerNames{i},4)  || ...
        strncmp('prob' ,layerNames{i},4)  || ...
        strncmp('fc'   ,layerNames{i},2) ...
        )
            layerNames{i} = [];
    end;
end;
layerNames = layerNames(~cellfun('isempty',layerNames));

%% Extract params
for layerIndex = 1:length(layerNames)
    layerName = layerNames{layerIndex};
    w = cnn.params(layerName,1).get_data();
    [K K C N] = size(w);
    
    
    %% Apply rounding
    w = round(scaleFactor*w);
    W = [W; w(:)];
end;

% Number of weights
t = [];
w = [];
w = W;
totalElements = numel(w);

% Number of Null Elements
zeroElements = numel(w(w==0));
oneElements = numel(w(w==1));

% Number of elements that are a power of 2
cmp = 0;
for i=1:numel(w)
    if ( w(i)~=0 && w(i)~=1 && mod(log2(abs(w(i))),1) ==0 )
        cmp = cmp +1;
        t = [t w(i)];
    end;
end;
p2Elements = cmp;

remainElements = totalElements - zeroElements - p2Elements;

Y = [zeroElements;oneElements;p2Elements;remainElements];
% labels = {'Null kernel','power of 2','remaining values' };
% pie([zeroElements./totalElements oneElements./totalElements p2Elements./totalElements]);
% legend(labels)
% title (strcat(modelName,' kernels rounded to  ', num2str(nBits),' bits'));

