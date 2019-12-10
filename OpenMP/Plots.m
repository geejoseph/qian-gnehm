% number of iterations is roughly log of largest dimension + 1
% since min number of unmerged regions halves with each iteration

% printing iterations adds about 0.5 ms for 9 iterations
% so roughly 0.06ms*n_iterations

% Lakeside.jpg is closest in size to their Lakeside, really same res?
% Their sequential time for sweeps on Lakeside is 16 ms, ours is 22 ms
% Comparing others our sequential sweeps seem slower in most cases and overall

% Their claim is sweep time goes sublinearly with image area
% Although total time goes linearly (makes sense with alloc no speedup)

% Beach, Cathedral of Learning, Einstein, Hamerschlag, Ilha Grande,
% Lakeside, Maracana, Morumbi, Museum, Sidney, World

% Time per iteration seq - time series for each image
par_is = [
    4.935299 3.046949 1.805421 1.146572 0.788929 0.599503 0.639653 0.661783 0.307965 0.126899 0 0 0;
    2.198928 1.372881 0.885769 0.623106 0.411576 0.431582 0.456556 0.321474 0.199614 0 0 0 0;
    7.715528 4.724867 2.636047 1.625163 1.141003 0.730985 0.792336 0.851346 0.593117 0.37620 0 0 0;
    10.877287 5.661698 3.261226 2.066964 1.364099 0.861523 0.870163 0.886900 0.636966 0.416027 0 0 0;
    5.507368 3.516103 5.197539 1.420144 0.911440 0.802080 0.814059 0.660438 0.528604 0.122770 0 0 0;
    2.483760 1.580318 1.031237 0.742735 0.477135 0.494016 0.521243 0.337963 0.207889 0 0 0 0;
    3.449176 2.413010 1.592409 0.970708 0.515139 0.540139 0.564968 0.583589 0.238975 0 0 0 0;
    2.487469 1.519378 1.016316 0.691760 0.439435 0.468606 0.499554 0.350389 0.199967 0 0 0 0;
    3.218421 1.930200 1.261297 0.819984 0.471609 0.501087 0.507389 0.553322 0.215320 0 0 0 0;
    31.431778 18.762138 10.453494 5.805401 3.122660 2.085521 1.200266 1.169052 1.246332 1.307543 0.769119 0 0;
    139.521435 66.981527 37.277906 21.911715 12.768824 7.177193 4.503860 2.882544 2.292318 2.442695 2.250561 2.171756 0.954452];
% hold off;
% plot(par_is(1,:));
% grid on;
% hold on;
% for i = 2:size(par_is, 1)-2
%     plot(par_is(i,:));
% end
% title('Time per iteration (ms), parallel, each time series is an image (closeup)');

% Time per iteration par - time series for each image
seq_is = [
    16.154250 11.386229 6.887309 4.129812 2.466255 1.514723 0.930561 0.512446 0.234187 0.089427 0 0 0;
    6.851269 5.035091 3.157101 1.945575 1.193195 0.711433 0.383616 0.201288 0.140793  0 0 0 0;
    25.189765 18.078572 11.086026 6.667377 4.041980 2.407697 1.448213 0.811225 0.412369 0.290652  0 0 0;
    31.514219 22.499138 13.833443 8.543149 5.103671 3.237475 1.755851 1.037082 0.445089 0.325739 0 0 0;
    17.974512 13.505537 8.724496 5.384623 3.283902 1.966870 1.119471 0.686752 0.372313 0.086329  0 0 0;
    7.520115 5.848203 3.766045 2.355391 1.355316 0.803715 0.471726 0.208179 0.148854 0 0 0 0;
    12.027249 9.954562 6.250025 3.623475 1.955416 1.151279 0.618859 0.326248 0.174339  0 0 0 0;
    7.324041 5.294740 3.262135 2.027408 1.177273 0.744513 0.470587 0.209300 0.14633 0 0 0 0;
    9.765489 8.089590 5.155012 2.974914 1.598726 0.907561 0.487659 0.308975 0.155282   0 0 0 0;
    110.695385 84.535206 55.383694 33.618740 19.060208 10.513052 5.671537 2.845193 1.561471 1.134878 0.557755 0 0;
    554.333193 388.623877 233.832772 138.466820 83.727628 58.934035 33.539929 19.242409 10.321669 5.913265 2.969576 1.636416 0.795684];
% hold off;
% plot(seq_is(1,:));
% grid on;
% hold on;
% for i = 2:size(seq_is, 1)
%     plot(seq_is(i,:));
% end
% title('Time per iteration (ms), sequential, each time series is an image');

% Time of first sweeps
% hold off;
% plot(seq_is(1,:));
% grid on;
% hold on;
% for i = 2:size(seq_is, 1)
%     plot(seq_is(i,:));
% end
% title('Time per iteration (ms), sequential, each time series is an image');

widths = [490 348 660 1024 1010 491 1633 491 491 491 5400];
lengths = [735 420 846 682 389 329 1632 491 330 398 2700];
sizes = widths .* lengths;

par_process_times = [18.035080 8.433232 26.706818 33.353597 23.807499 9.664202 94.637913 12.994019 9.382310 11.266937 400.295802];
par_sweep_times = [14.110508 6.944003 21.239481 26.960009 19.529408 7.920911 77.436376 10.914617 7.716722 9.526094 303.244673];
par_update_times = [3.258488 1.204334 4.350397 4.964905 3.528256 1.395978 12.583388 1.576577 1.365790 1.370015 63.303502];

seq_process_times = [58.343468 24.862990 91.977909 114.606009 71.795913 28.061843 404.379453 43.352443 26.403808 35.312851 2144.478375];
seq_sweep_times = [44.343994 19.657728 70.478758 88.337734 53.144927 22.516143 325.649460 36.122618 20.688940 29.483823 1532.425980];
seq_update_times = [13.330727 4.917434 20.411259 24.811199 17.844410 5.200592 74.128837 6.726883 5.411604 5.412038 578.555428];

total_speedups = seq_process_times ./ par_process_times;
sweep_speedups = seq_sweep_times ./ par_sweep_times;
update_speedups = seq_update_times ./ par_update_times;

% Plot with total size vs. process time par and seq
% hold on;
% scatter(seq_process_times(:,1:end-1), sizes(:,1:end-1));
% scatter(par_process_times(:,1:end-1), sizes(:,1:end-1));
% legend('sequential', 'parallel');
% title('Total process times (ms) vs. image size (px), sequential and parallel');

% Plot with size vs. sweep time par and seq
% hold on;
% scatter(seq_sweep_times(:,1:end-1), sizes(:,1:end-1));
% scatter(par_sweep_times(:,1:end-1), sizes(:,1:end-1));
% legend('sequential', 'parallel');
% title('Sweep times (ms) vs. image size (px), sequential and parallel');

% Plot with size vs. sweep speedup
% hold on;
% scatter(sizes, sweep_speedups);
% title('Image size (px) vs. sweep speedup');

% Plot with size vs. update time par and seq
% hold on;
% scatter(seq_update_times(:,1:end-1), sizes(:,1:end-1));
% scatter(par_update_times(:,1:end-1), sizes(:,1:end-1));
% legend('sequential', 'parallel');
% xlim([0 350])
% title('Update times (ms) vs. image size (px), sequential and parallel');

% Plot with size vs. update speedup
% hold on;
% scatter(sizes, update_speedups);
% title('Image size (px) vs. update speedup');

% Plot with size vs. total speedup
% hold on;
% scatter(sizes, total_speedups);
% title('Image size (px) vs. total speedup');

second_sweep_times = [3.384049 0.081145 3.907290 4.013704;
    1.479837 0.053550 1.689778 1.811892;
    5.385144 0.108122 6.158849 6.426398;
    6.765443 0.178256 7.598872 7.956521;
    4.019015 0.166033 4.515385 4.804998;
    1.750313 0.080388 1.926185 2.091275;
    25.894199 0.296742 27.896552 30.447665;
    2.960735 0.099192 3.148935 3.745659;
    1.585570 0.075236 1.776699 1.857194;
    2.416302 0.096540 2.560309 3.016389;
     116.263889 0.899999 133.484005 137.975881]';

% Plot with size vs. each sweep
hold on;
scatter(sizes(:,1:end-1), second_sweep_times(1,1:end-1));
scatter(sizes(:,1:end-1), second_sweep_times(2,1:end-1));
scatter(sizes(:,1:end-1), second_sweep_times(3,1:end-1));
scatter(sizes(:,1:end-1), second_sweep_times(4,1:end-1));
legend('horizontal sweep', 'first diagonal sweep', 'vertical sweep', ...
'second horizontal sweep');
title('Image size (px) vs. individual sweep times (ms) in second iteration');