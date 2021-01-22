clear all;
close all;

gpu1 = load('lib_gpu1_1thread.out');
aos = load('lib_gpu2.out');

figure(1);
x=gpu1(:,2);
y=gpu1(:,3:4);
hAxes = loglog(x,y);
% set(gca, 'Xscale', 'log');
% set(gca, 'Yscale', 'log');
%plot(aos(:,1),aos(:,2:4));
legend('CBLAS 1 thread', 'GPU 1st version' );
xlabel('Memory footprint (kB)');
ylabel('Performance (Mflops/s)');
title("Performance of first gpu version and CPU CBLAS single threaded.");
ylim(hAxes,[0, 1e6]);