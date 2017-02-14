clc;
clear all;
close all;

k = [9;25;49;121];
alm = [22;75;146;331];

% scatter(k,alm)
X = [ones(length(k),1) k];
b = X\alm;
almCalc = X*b;
abs(sum(almCalc - alm))
scatter(k,alm)
hold on
plot(k,almCalc,'--')