costMatrix = importdata('multCost_data_7bits.mat');
x = costMatrix(:,1)./ max(costMatrix(:,1));
y = costMatrix(:,2);
scatter(x,y)
axis([-1 1 0 20]);

