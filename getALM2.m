function J = getALM2(fitFilename)
%     fitFilename = '/home/kamel/Seafile/Kamel/Reports/ce_alexNet.fit.rpt';
    fitFile     = fopen(fitFilename,'r');
    macMarker = ';       |convElement:\CEs_loop:';
    sumMarker = ';          |parallel_add:parallel_add_component|';
    J = [];
    J_macs = [];
    J_sums = [];
    instNums = [];

    while ~feof(fitFile)
        line = fgetl(fitFile);
        if (~isempty(line))
            if (strncmp(macMarker,line,length(macMarker)));
                splitedLine = strsplit(line);
                
                instNum     = str2num(splitedLine{3});
                instNums = [instNums;instNum];
                
                J_mac       = str2num(splitedLine{6});
                J_macs   = [J_macs ; J_mac];
            end;
            
            if (strncmp(sumMarker,line,length(sumMarker)));
                splitedLine = strsplit(line);
                J_sum       = str2num(splitedLine{4});
                J_sums      = [J_sums ; J_sum];
            end;            
            
        end;
    end;
    J = [instNums J_macs J_sums];
        
