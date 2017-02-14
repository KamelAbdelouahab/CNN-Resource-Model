clc;
close all;
clear all;

nBits = 8;
scaleFactor = 2 ^ (nBits-1) - 1;

% CNN
protoFile = '/home/kamel/Seafile/Kamel/alexNet_compressed/caffe/alexNet.prototxt';
modelFile = '/home/kamel/Seafile/Kamel/alexNet_compressed/caffe/alexNet_compressed.caffemodel';
cnn       = caffe.Net(protoFile,modelFile,'test');

% Static for now, not used to matCaffe
W1 = cnn.params('conv1',1).get_data();
W2 = cnn.params('conv2',1).get_data();
W3 = cnn.params('conv3',1).get_data();
W4 = cnn.params('conv4',1).get_data();
W5 = cnn.params('conv5',1).get_data();
