function cost = multCost(roundedValue,nBits)
    fileName = strcat('multCost_data_',num2str(nBits),'bits.mat');
%     load ./multCost_data.mat 
    costMatrix = importdata(fileName);
    for i = 1:length(costMatrix);
        if roundedValue==costMatrix(i,1);
            tmpCost = costMatrix(i,2);
            if (tmpCost<1)
                cost = 0;
            else
                cost = tmpCost;
            end;
        end;
    end;
end
