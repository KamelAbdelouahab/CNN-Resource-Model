clear all;
close all;
clc;

fileName = 'multCost_data.mat'
load(fileName)

values = costMult_data(:,1);
alms   = costMult_data(:,2);
aluts  = costMult_data(:,3);
scatter(values,aluts)

% for i=1:length(aluts);
%     if aluts(i) <= 1
%         aluts(i) = 0;
%     end
% end

costMult_data(:,3) = aluts;
save(fileName,'costMult_data');
