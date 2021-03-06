% Julia apparently is incapable of plotting on my computer, so use
% MATLAB instead...

% derivation_path = '../../derivation/';
mat_path = '../mat/';

load(strcat(mat_path, 'numexp__fmm_speed.mat'));

semilogy(Ns, directtimes);
hold on;
semilogy(Ns, fmmtimes, '--');
% xlabel('N', 'interpreter', 'latex');
% ylabel('Time (sec.)', 'interpreter', 'latex');
legend('Direct', 'MLFMM', 'Location', 'southeast');
set(gca, 'color', 'none');
save_as_eps(gcf, 'fmm_speed.eps', 150, 850, 400);

load(strcat(mat_path, 'numexp__test_series.mat'));

figure();
for ii = 1:size(semicircle, 2)
    plot(X, semicircle(:, ii));
    hold on;
end
xlim([0 2*pi]);
title('Semicircle', 'interpreter', 'latex');
xlabel('$x$', 'interpreter', 'latex');
ylabel('$f(x)$', 'interpreter', 'latex');
set(gca, 'XTick', [0, pi/2, pi, 3*pi/2, 2*pi]);
set(gca, 'XTickLabel', []);
save_as_eps(gcf, 'semicircle.eps', 150, 450, 350);

figure();
for ii = 1:size(triangle, 2)
    plot(X, triangle(:, ii));
    hold on;
end
xlim([0 2*pi]);
title('Triangle', 'interpreter', 'latex');
xlabel('$x$', 'interpreter', 'latex');
ylabel('$f(x)$', 'interpreter', 'latex');
set(gca, 'XTick', [0, pi/2, pi, 3*pi/2, 2*pi]);
set(gca, 'XTickLabel', []);
save_as_eps(gcf, 'triangle.eps', 150, 450, 350);

figure();
for ii = 1:size(sawtooth, 2)
    plot(X, sawtooth(:, ii));
    hold on;
end
xlim([0 2*pi]);
title('Sawtooth', 'interpreter', 'latex');
xlabel('$x$', 'interpreter', 'latex');
ylabel('$f(x)$', 'interpreter', 'latex');
set(gca, 'XTick', [0, pi/2, pi, 3*pi/2, 2*pi]);
set(gca, 'XTickLabel', []);
save_as_eps(gcf, 'sawtooth.eps', 150, 450, 350);

figure();
for ii = 1:size(square, 2)
    plot(X, square(:, ii));
    hold on;
end
xlim([0 2*pi]);
title('Square', 'interpreter', 'latex');
xlabel('$x$', 'interpreter', 'latex');
ylabel('$f(x)$', 'interpreter', 'latex');
set(gca, 'XTick', [0, pi/2, pi, 3*pi/2, 2*pi]);
set(gca, 'XTickLabel', []);
save_as_eps(gcf, 'square.eps', 150, 450, 350);

load(strcat(mat_path, 'numexp__interp_speed.mat'));

figure();
semilogy(Ks, groundtruth_time_avg);
hold on;
semilogy(Ks, persum_time_avg, '--');
xlabel('K', 'interpreter', 'latex');
ylabel('Time (sec.)', 'interpreter', 'latex');
legend('Ground Truth', 'Periodic Sum', 'Location', 'southeast');
xlim([min(Ks) max(Ks)])
save_as_eps(gcf, 'interp_speed.eps', 150, 850, 400);

exit;