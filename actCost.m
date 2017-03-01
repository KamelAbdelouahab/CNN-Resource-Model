function alm = actCost(channels)
%     model = [42.6322; 6.5494];
    model  = [3.639; 2.3349 ];
    channels = [ones(length(channels),1) channels];
    alm = channels*model;
