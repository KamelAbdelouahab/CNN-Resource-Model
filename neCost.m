function c = neCost(k)
    switch k
        case 3
            c = 87;
        case 5
            c = 181;
        case 7
            c = 309;
        case 11
            c = 1154;
        otherwise
            msg = 'Error, kernel should have a size of 3,5,7 or 11 only';
            error(msg);
    % end;
end;
