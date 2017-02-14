% function J_sum = sumCost(k)
%     Model = [4.5833; 2.7239];
%     X = [ones(length(k),1) k];
%     J_sum = X*Model;

function y = sumCost(x)
    switch k
        case 3
            c = 87;
        case 5
            c = 26.24;
        case 7
            c = 309;
        case 11
            c = 47;
        otherwise
            msg = 'Error, kernel should have a size of 3,5,7 or 11 only';
            error(msg);
    % end;
end;
