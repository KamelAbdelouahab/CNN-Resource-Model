function [ y1 y2] = alm_metric( w )
y1 = 0;
y2 = 0;
w = w(:);

% Number of null kernels
y1 = 100 .* (numel(w) - nnz(w)) ./ numel(w);

% Number of pow2 kernels

w2p = w;
for i=1:length(w)
    if ( w2p(i) > 0 && mod(log2(abs(w2p(i))),1) ==0 )
        w2p(i) = 0;
    end;
end;
y2 = 100 .* (numel(w2p) - nnz(w2p)) ./ numel(w2p);
end

