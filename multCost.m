function y = multCost(x)
    load ./multCost_data.mat 
    for i = 1:length(costMult_data);
        if x==costMult_data(i,1);
            r = costMult_data(i,2);
            if r<1
                y = 0;
            else
                y = r;
            end;
        end;

    end;
end
