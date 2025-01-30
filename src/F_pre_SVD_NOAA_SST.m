function [U, S, V, Xorg, meansst, n, t] = F_pre_SVD_NOAA_SST(m, time, mask, sst)

    t0   = datenum('1-Jan-1800 00:00:00') + time(1);
    tfin = datenum('1-Jan-1800 00:00:00') + time(end);
    datestr(t0,   'yyyy/mm/dd');
    datestr(tfin, 'yyyy/mm/dd');

    [mm,nn, ~] = size(sst);
    n_sst = mm * nn;
    snapshot = reshape(sst(:, :, 1:length(time)), n_sst, length(time));
    Xall = snapshot(mask==1, :);

    Iord = 1:m;%length(time); %Iord = 1:52*20;
    rng(1);
    Itrain = Iord;
    Xtrain = Xall(:, Itrain);
    
    meansst = mean(Xtrain, 2);
    Xtrain = Xtrain - meansst;
%     Xtrain = bsxfun(@minus, Xtrain, meansst);

    tic
    [U, S, V] = svd(Xtrain, 'econ');
    t = toc;
    Xorg = U * S * V';
    n = size(U, 1);

end 