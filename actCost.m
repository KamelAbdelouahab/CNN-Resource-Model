function alm = actCost(channels)
    model = [42.6322; 6.5494];
    channels = [ones(length(channels),1) channels];
    alm = channels*model;
