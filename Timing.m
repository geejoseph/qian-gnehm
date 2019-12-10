% number of iterations is roughly log of largest dimension + 1
% since min number of unmerged regions halves with each iteration

% printing iterations adds about 0.5 ms for 9 iterations
% so roughly 0.06ms*n_iterations

% Lakeside.jpg is closest in size to their Lakeside, really same res?
% Their sequential time for sweeps on Lakeside is 16 ms, ours is 22 ms
% Comparing others our sequential sweeps seem slower in most cases and overall

% Their claim is sweep time goes sublinearly with image area
% Although total time goes linearly (makes sense with alloc no speedup)

% Time per iteration seq - time series for each image with label of name and
% size

% Time of first sweeps

% Plot with total size vs. process time par and seq
% 2d plot with dimensions vs. process time par and seq

% Plot with size vs. sweep time
% 2d plot with dimensions vs. sweep time

% Plot with size vs. update time par and seq

% Plot with size vs. speedup
% 2d plot with dimensions vs. speedup