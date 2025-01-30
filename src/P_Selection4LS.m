%% Main program

clear
close all

%% Parameters =======================================================
flag = 0;
N = 10000;
N_CV = 5; % Number of K-fold cross-validation
M0 = 6250;
Ms2 = [10, 20, 50, 100, 200, 500, 1e3, 2e3, 5e3];
Ms = round(N_CV / (N_CV - 1) * Ms2);
lambda0 = 0;
lambdas = [0, 10.^(-12:-4)];
beta = 0.9999;
r0 = 10;
rs = [1, 2, 5, 10, 20, 50, 100];
Ny0 = 10;
Nys = [1, 2, 5, 10, 20, 50, 100];
sigma0 = 1e-3;
sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0];
p0 = 10;
ParentDir = '../data/resultsLS';
mkdir(ParentDir);

%% Preprocessing ================================
alpha0 = (1 - beta) ^ (1 / (2 * r0));
alphas = (1 - beta) .^ (1 ./ (2 * rs));
r0_true = log(1 - beta * (1 - alpha0^(2 * N))) / (2 * log(alpha0));
rs_true = log(1 - beta * (1 - alphas.^(2 * N))) ./ (2 * log(alphas));
SN = 1 / (1 - alpha0^2) / (N * sigma0^2);

%% Data generation ================================================
rng(0)

Bs = cell(length(Nys), 1);
for i = 1:length(Nys)
    B = randn(Nys(i), N);
    [B, ~] = qr(B', 'econ');
    Bs{i} = B';
end
B0 = Bs{Nys == Ny0};

C = randn(N, N);
[C, ~] = qr(C, 'econ');

kappa0 = alpha0 .^ (2 * (0:N-1)');
Z0 = kappa0 .* randn(N, M0);
X0 = C * Z0 + sigma0 * randn(N, M0);
Y0 = B0 * Z0;

%% Parameter study on r ==================================
for i = 1:length(alphas)
    alpha = alphas(i);
    r = rs(i);
    fprintf('r = %d \n', r)
    
    kappa = alpha .^ (2 * (0:N-1)');
    Z = kappa .* randn(N, M0);
    X = C * Z + sigma0 * randn(N, M0);
    Y = B0 * Z;
    
    [Itrain, Itest] = F_RandCVDivide(size(X, 2), N_CV);
    for j = 1:N_CV
        itrain = Itrain{j};
        itest  = Itest {j};
        
        [Xtrain, Ytrain, Xtest, Ytest, U, Sigma, tSVD] = ...
            F_DataPreprocessing4Estimation(X, Y, r, itrain, itest);
        
        [errors, times] = F_selection2(Xtrain, Ytrain, Xtest, Ytest, ...
            U, Sigma, r, p0, lambda0, lambda0, tSVD);

        if i == 1 && j == 1
            sels = keys(errors);
            error_mat = zeros(length(alphas), N_CV, length(sels));
            time_mat = zeros(length(alphas), N_CV, length(sels));
        end

        for k = 1:length(sels)
            error_mat(i, j, k) = errors(sels(k));
            time_mat(i, j, k) = times(sels(k));
        end
    end
end

save([ParentDir, '/error_alpha'], 'error_mat', 'time_mat', ...
    'sels', 'alphas', 'rs')

%% Parameter study on Ny ==================================
for i = 1:length(Nys)
    Ny = Nys(i);
    fprintf('Ny = %d \n', Ny)
    
    Z = Z0;
    X = X0;
    Y = Bs{i} * Z;

    [Itrain, Itest] = F_RandCVDivide(size(X, 2), N_CV);
    for j = 1:N_CV
        itrain = Itrain{j};
        itest  = Itest {j};
        
        [Xtrain, Ytrain, Xtest, Ytest, U, Sigma, tSVD] = ...
            F_DataPreprocessing4Estimation(X, Y, r0, itrain, itest);
        
        [errors, times] = F_selection2(Xtrain, Ytrain, Xtest, Ytest, ...
            U, Sigma, r0, p0, lambda0, lambda0, tSVD);

        if i == 1 && j == 1
            sels = keys(errors);
            error_mat = zeros(length(Nys), N_CV, length(sels));
            time_mat = zeros(length(Nys), N_CV, length(sels));
        end

        for k = 1:length(sels)
            error_mat(i, j, k) = errors(sels(k));
            time_mat(i, j, k) = times(sels(k));
        end
    end
end

save([ParentDir, '/error_Ny'], 'error_mat', 'time_mat', 'sels', 'Nys')


%% Parameter study on sigma ==================================
for i = 1:length(sigmas)
    sigma = sigmas(i);
    fprintf('sigma = %e \n', sigma)
    
    Z = Z0;
    X = C * Z + sigma * randn(N, M0);
    Y = Y0;

    [Itrain, Itest] = F_RandCVDivide(size(X, 2), N_CV);
    for j = 1:N_CV
        itrain = Itrain{j};
        itest  = Itest {j};
        
        [Xtrain, Ytrain, Xtest, Ytest, U, Sigma, tSVD] = ...
            F_DataPreprocessing4Estimation(X, Y, r0, itrain, itest);
        
        [errors, times] = F_selection2(Xtrain, Ytrain, Xtest, Ytest, ...
            U, Sigma, r0, p0, lambda0, lambda0, tSVD);

        if i == 1 && j == 1
            sels = keys(errors);
            error_mat = zeros(length(sigmas), N_CV, length(sels));
            time_mat = zeros(length(sigmas), N_CV, length(sels));
        end

        for k = 1:length(sels)
            error_mat(i, j, k) = errors(sels(k));
            time_mat(i, j, k) = times(sels(k));
        end
    end
end

save([ParentDir, '/error_sigma'], 'error_mat', 'time_mat', ...
    'sels', 'sigmas')


%% Parameter study on M ==================================
for i = 1:length(Ms)
    M = Ms(i);
    fprintf('M = %e \n', M)
    
    kappa = alpha0 .^ (2 * (0:N-1)');
    Z = kappa .* randn(N, M);
    X = C * Z + sigma0 * randn(N, M);
    Y = B0 * Z;

    [Itrain, Itest] = F_RandCVDivide(size(X, 2), N_CV);

    for j = 1:N_CV
        itrain = Itrain{j};
        itest  = Itest {j};
        
        [Xtrain, Ytrain, Xtest, Ytest, U, Sigma, tSVD] = ...
            F_DataPreprocessing4Estimation(X, Y, r0, itrain, itest);
        
        [errors, times] = F_selection2(Xtrain, Ytrain, Xtest, Ytest, ...
            U, Sigma, r0, p0, lambda0, lambda0, tSVD);

        if i == 1 && j == 1
            sels = keys(errors);
            error_mat = zeros(length(Ms), N_CV, length(sels));
            time_mat = zeros(length(Ms), N_CV, length(sels));
        end

        for k = 1:length(sels)
            error_mat(i, j, k) = errors(sels(k));
            time_mat(i, j, k) = times(sels(k));
        end
    end
end

save([ParentDir, '/error_M'], 'error_mat', 'time_mat', ...
    'sels', 'Ms2')

%% Parameter study on lambdas ==================================
for i = 1:length(Ms)
    M = Ms(i);
    fprintf('M = %d \n', M)
    
    kappa = alpha0 .^ (2 * (0:N-1)');
    Z = kappa .* randn(N, M);
    X = C * Z + sigma0 * randn(N, M);
    Y = B0 * Z;

    [Itrain, Itest] = F_RandCVDivide(size(X, 2), N_CV);

    for j = 1:N_CV
        itrain = Itrain{j};
        itest  = Itest {j};
    
        [Xtrain, Ytrain, Xtest, Ytest] = ...
            F_DataPreprocessing4Estimation2(X, Y, itrain, itest);
    
        errors = F_selection_lambda(Xtrain, Ytrain, Xtest, Ytest, ...
            p0, lambdas, lambdas);

        if i == 1 && j == 1
            error_mat = zeros(length(Ms), N_CV, length(lambdas));
        end

        for k = 1:length(lambdas)
            error_mat(i, j, k) = errors(k);
        end
    end
end

save([ParentDir, '/error_lambda_M'], ...
    'error_mat', 'lambdas', 'Ms2')


%% Parameter study on lambdas ==================================
for i = 1:length(Ms)
    M = Ms(i);
    fprintf('M = %d \n', M)
    
    kappa = alpha0 .^ (2 * (0:N-1)');
    Z = kappa .* randn(N, M);
    X = C * Z + sigma0 * randn(N, M);
    Y = B0 * Z;

    [Itrain, Itest] = F_RandCVDivide(size(X, 2), N_CV);

    for j = 1:N_CV
        itrain = Itrain{j};
        itest  = Itest {j};
    
        [Xtrain, Ytrain, Xtest, Ytest] = ...
            F_DataPreprocessing4Estimation2(X, Y, itrain, itest);
    
        errors = F_selection_lambda(Xtrain, Ytrain, Xtest, Ytest, ...
            p0, 0*lambdas, lambdas);

        if i == 1 && j == 1
            error_mat = zeros(length(Ms), N_CV, length(lambdas));
        end

        for k = 1:length(lambdas)
            error_mat(i, j, k) = errors(k);
        end
    end
end

save([ParentDir, '/error_lambda0_M'], ...
    'error_mat', 'lambdas', 'Ms2')

%% Plot CV results
clear

blue = [0; 83; 168] / 255;
red  = [192; 0; 0] / 255;
green = [0; 110; 79] / 255;
purple = [102; 0; 102] / 255;
yellow = [246; 191; 0] / 255;
grey = 0.2 * [1; 1; 1];
Color = {blue, red, green, purple, yellow};

white = [1; 1; 1];
alpha = 0.25;

width = 12;
height = 11;
FontSize = 13;
FontName = 'Times New Roman';
LineWidth = 1;

InputParentDir = '../data/resultsLS';
mkdir('../pngs')
mkdir('../pdfs')
OutputBase1 = '../pngs/LS_';
OutputBase2 = '../pdfs/LS_';

%% plot r vs error
Dir = InputParentDir;
load([Dir, '/error_alpha'], 'error_mat', 'rs', 'sels')
H = gobjects(1, length(sels));
sels(1) = "GREG $$ (\tilde{\lambda} = 0) $$";

figure
hold on
error_mat = error_mat ./ error_mat(:, :, 1);
for i = 1:length(sels)
    h = F_errorfill_meanstd(rs', error_mat(:, :, i));
    h.FaceColor = Color{i};
    h.FaceAlpha = alpha;
    h.EdgeColor = 'none';
end
for i = 1:length(sels)
    h = F_plot_mean(rs', error_mat(:, :, i));
    h.Color = Color{i};
    h.LineWidth = LineWidth;
    h.DisplayName = sels(i);
    H(i) = h;
end
lgd = legend(H);
lgd.NumColumns = 1;
lgd.Location = "NorthWest";
lgd.Interpreter = 'Latex';
xlabel('Statistical dimension, $$ \hat{r} $$', 'Interpreter', 'Latex')
ylabel('Normalized estimation error', 'Interpreter', 'Latex')
grid on
ax = gca;
ax.Box = 'on';
ax.FontSize = FontSize;
ax.FontName = FontName;
ax.XScale = 'log';
% ax.YScale = 'log';
F_change_size(gcf, width, height)
%     legend(names)
hold off
Name = 'CV_error_meanstd_alpha_norm';
FileName1 = [OutputBase1, Name, '.png'];
FileName2 = [OutputBase2, Name, '.pdf'];
delete(FileName1)
exportgraphics(gca, FileName1)
delete(FileName2)
exportgraphics(gca, FileName2)


%% plot Ny vs error
Dir = InputParentDir;
load([Dir, '/error_Ny'], 'error_mat', 'Nys', 'sels')
H = gobjects(1, length(sels));
sels(1) = "GREG $$ (\tilde{\lambda} = 0) $$";

figure
error_mat = error_mat ./ error_mat(:, :, 1);
hold on
for i = 1:length(sels)
    h = F_errorfill_meanstd(Nys', error_mat(:, :, i));
    h.FaceColor = Color{i};
    h.FaceAlpha = alpha;
    h.EdgeColor = 'none';
end
for i = 1:length(sels)
    h = F_plot_mean(Nys', error_mat(:, :, i));
    h.Color = Color{i};
    h.LineWidth = LineWidth;
    h.DisplayName = sels(i);
    H(i) = h;
end
lgd = legend(H);
lgd.NumColumns = 1;
lgd.Location = "NorthEast";
lgd.Interpreter = 'Latex';
xlabel('Number of targets, $$ N_y $$', 'Interpreter', 'Latex')
ylabel('Normalized estimation error', 'Interpreter', 'Latex')
grid on
ax = gca;
ax.Box = 'on';
ax.FontSize = FontSize;
ax.FontName = FontName;
ax.XScale = 'log';
% ax.YScale = 'log';
% ylim([10^(-4), inf])
F_change_size(gcf, width, height)
%     legend(names)
hold off
Name = 'CV_error_meanstd_Ny_norm';
FileName1 = [OutputBase1, Name, '.png'];
FileName2 = [OutputBase2, Name, '.pdf'];
delete(FileName1)
exportgraphics(gca, FileName1)
delete(FileName2)
exportgraphics(gca, FileName2)


%% plot sigma vs error
Dir = InputParentDir;
load([Dir, '/error_sigma'], 'error_mat', 'sigmas', 'sels')
H = gobjects(1, length(sels));
sels(1) = "GREG $$ (\tilde{\lambda} = 0) $$";

figure
error_mat = error_mat ./ error_mat(:, :, 1);
hold on
for i = 1:length(sels)
    h = F_errorfill_meanstd(sigmas', error_mat(:, :, i));
    h.FaceColor = Color{i};
    h.FaceAlpha = alpha;
    h.EdgeColor = 'none';
end
for i = 1:length(sels)
    h = F_plot_mean(sigmas', error_mat(:, :, i));
    h.Color = Color{i};
    h.LineWidth = LineWidth;
    h.DisplayName = sels(i);
    H(i) = h;
end
lgd = legend(H);
lgd.NumColumns = 1;
lgd.Location = 'NorthEast';
lgd.Interpreter = 'Latex';
xlabel('Noise STD., $$ \sigma_v $$', 'Interpreter', 'Latex')
ylabel('Normalized estimation error', 'Interpreter', 'Latex')
grid on
ax = gca;
ax.Box = 'on';
ax.FontSize = FontSize;
ax.FontName = FontName;
ax.XScale = 'log';
% ax.YScale = 'log';
ylim([-inf, inf])
xlim([sigmas(1), sigmas(end)])
% xticks(sigmas(1:2:end))
F_change_size(gcf, width, height)
%     legend(names)
hold off
Name = 'CV_error_meanstd_sigma_norm';
FileName1 = [OutputBase1, Name, '.png'];
FileName2 = [OutputBase2, Name, '.pdf'];
delete(FileName1)
exportgraphics(gca, FileName1)
delete(FileName2)
exportgraphics(gca, FileName2)

%% plot lambda, M vs error
Dir = InputParentDir;
load([Dir, '/error_lambda_M'], 'error_mat', 'Ms2', 'lambdas')

lambdas(1) = 0.1 * lambdas(2);
error_mean = mean(error_mat, 2);
error_mean = squeeze(error_mean);
X = Ms2' * ones(1, length(lambdas));
Y = ones(length(Ms2), 1) * lambdas;
[min_error_mean, imin] = min(error_mean, [], 2);
Xmin = diag(X(:, imin));
Ymin = diag(Y(:, imin));

figure
hold on
error_mean = error_mean ./ error_mean(:, 1);
s = scatter(X(:), Y(:), [], error_mean(:), 'filled');
s2 = scatter(Xmin(:), Ymin(:));
s2.SizeData = 1.3 * s.SizeData;
s2.MarkerEdgeColor = 'k';
s2.LineWidth = 1;
% s.MarkerFaceColor = 'auto';
xlabel('Number of snapshots, $$ M $$', 'Interpreter', 'Latex')
ylabel('Regularization, $$ \lambda $$', 'Interpreter', 'Latex')
grid on
ax = gca;
ax.Box = 'on';
ax.FontSize = FontSize;
ax.FontName = FontName;
ax.XScale = 'log';
ax.YScale = 'log';
ax.YTick = [lambdas(1), lambdas(2:2:end)];
YTickLabel = ax.YTickLabel;
YTickLabel{1} = '0';
ax.YTickLabel = YTickLabel;
% ax.ColorScale = 'log';
xlim([-inf, Ms2(end)])
ylim([lambdas(1), inf])
clim([-inf, 1.1])
F_change_size(gcf, width, height)
hold off
colorbar
cmap1 = colormap('turbo');
cmap1(1:20, :) = [];
cmap1(end-19:end, :) = [];
colormap(cmap1)
Name = 'CV_error_scatter_M_norm';
FileName1 = [OutputBase1, Name, '.png'];
FileName2 = [OutputBase2, Name, '.pdf'];
delete(FileName1)
exportgraphics(gca, FileName1)
delete(FileName2)
exportgraphics(gca, FileName2)


%% plot effect of lambda on selection
Dir = InputParentDir;

load([Dir, '/error_lambda_M'], 'error_mat', 'Ms2', 'lambdas')
error_mean = mean(error_mat, 2);
error_mean = squeeze(error_mean);

load([Dir, '/error_lambda0_M'], 'error_mat')
error0_mean = mean(error_mat, 2);
error0_mean = squeeze(error0_mean);

lambdas(1) = 0.1 * lambdas(2);
X = Ms2' * ones(1, length(lambdas));
Y = ones(length(Ms2), 1) * lambdas;
[min_error_mean, imin] = min(error_mean, [], 2);
Xmin = diag(X(:, imin));
Ymin = diag(Y(:, imin));

error_mean = (error0_mean - error_mean) ./ error0_mean;

figure
hold on
s = scatter(X(:), Y(:), [], error_mean(:), 'filled');
s2 = scatter(Xmin(:), Ymin(:));
s2.SizeData = 1.3 * s.SizeData;
s2.MarkerEdgeColor = 'k';
s2.LineWidth = 1;
% s.MarkerFaceColor = 'auto';
xlabel('Number of snapshots, $$ M $$', 'Interpreter', 'Latex')
ylabel('Regularization, $$ \tilde{\lambda} $$', 'Interpreter', 'Latex')
grid on
ax = gca;
ax.Box = 'on';
ax.FontSize = FontSize;
ax.FontName = FontName;
ax.XScale = 'log';
ax.YScale = 'log';
ax.YTick = [lambdas(1), lambdas(2:2:end)];
YTickLabel = ax.YTickLabel;
YTickLabel{1} = '0';
ax.YTickLabel = YTickLabel;
% ax.ColorScale = 'log';
xlim([-inf, Ms2(end)])
ylim([lambdas(1), inf])
% clim([-inf, 0.14])
F_change_size(gcf, width, height)
hold off
colorbar
% cmap1 = cmap_br();
cmap1 = flipud(hot);
cmap1(end-20:end, :) = [];
colormap(cmap1)
Name = 'CV_error_scatter_M_sel';
FileName1 = [OutputBase1, Name, '.png'];
FileName2 = [OutputBase2, Name, '.pdf'];
delete(FileName1)
exportgraphics(gca, FileName1)
delete(FileName2)
exportgraphics(gca, FileName2)


%% plot M vs time
Dir = InputParentDir;
load([Dir, '/error_M'], 'time_mat', 'Ms2', 'sels')
H = gobjects(1, length(sels));
sels(1) = "GREG $$ (\tilde{\lambda} = 0) $$";

figure
hold on
for i = 1:length(sels)
    h = F_errorfill_meanstd(Ms2', time_mat(:, :, i));
    h.FaceColor = Color{i};
    h.FaceAlpha = alpha;
    h.EdgeColor = 'none';
end
for i = 1:length(sels)
    h = F_plot_mean(Ms2', time_mat(:, :, i));
    h.Color = Color{i};
    h.LineWidth = LineWidth;
    h.DisplayName = sels(i);
    H(i) = h;
end
lgd = legend(H);
lgd.NumColumns = 1;
lgd.Location = 'NorthWest';
lgd.Interpreter = 'Latex';
xlabel('Number of snapshots, $$ M $$', 'Interpreter', 'Latex')
ylabel('Time [s]', 'Interpreter', 'Latex')
grid on
ax = gca;
ax.Box = 'on';
ax.FontSize = FontSize;
ax.FontName = FontName;
ax.XScale = 'log';
ax.YScale = 'log';
% ylim([10^(-4), inf])
xlim([-inf, inf])
F_change_size(gcf, width, height)
%     legend(names)
hold off
Name = 'CV_time_meanstd_M';
FileName1 = [OutputBase1, Name, '.png'];
FileName2 = [OutputBase2, Name, '.pdf'];
delete(FileName1)
exportgraphics(gca, FileName1)
delete(FileName2)
exportgraphics(gca, FileName2)

function [Xtrain, Ytrain, Xtest, Ytest, U, Sigma, tSVD, eSVD, ESVD] = ...
    F_DataPreprocessing4Estimation(X, Y, r, itrain, itest)

Xtrain = X(:, itrain);
Ytrain = Y(:, itrain);
tic;
[U, Sigma, ~] = svd(Xtrain, 'econ');
% [U, Sigma, ~] = svds(Xtrain, r);
tSVD = toc;
normX2 = norm(Xtrain(:))^2;
ESVD = normX2 - sum(diag(Sigma(1:r, 1:r)) .^ 2);
eSVD = sqrt(ESVD / normX2);
Xtest = X(:, itest);
Ytest = Y(:, itest);

end

function [Xtrain, Ytrain, Xtest, Ytest] = ...
    F_DataPreprocessing4Estimation2(X, Y, itrain, itest)

Xtrain = X(:, itrain);
Ytrain = Y(:, itrain);
Xtest = X(:, itest);
Ytest = Y(:, itest);

end

function [errors, times] = F_selection2(Xtrain, Ytrain, Xtest, Ytest, ...
    U, Sigma, r, p, lambda, lambda2, tSVD)

errors = dictionary;
times = dictionary;


%% GREG -----------------------------------------------------------
disp([num2str(p), ' sensors are calcuratiing with GREG']);

[S, time] = F_GREG(Xtrain, Ytrain, p, lambda);
error = F_LinearEstimation(Xtrain, Ytrain, Xtest, Ytest, ...
    S, lambda2, 'NMSE');
errors('GREG') = error;
times('GREG') = time(end);

%% REG -----------------------------------------------------------
disp([num2str(p), ' sensors are calcuratiing with REG']);

[S, time] = F_REG(Xtrain, p);
error = F_LinearEstimation(Xtrain, Ytrain, Xtest, Ytest, ...
    S, lambda2, 'NMSE');
errors('REG') = error;
times('REG') = time(end);

%% OMP -----------------------------------------------------------
disp([num2str(p), ' sensors are calcuratiing with SOMP']);

[S, time] = F_SOMP(Xtrain, Ytrain, p);
error = F_LinearEstimation(Xtrain, Ytrain, Xtest, Ytest, ...
    S, lambda2, 'NMSE');
errors('SOMP') = error;
times('SOMP') = time(end);

%% QD ------------------------------------------------------------
disp([num2str(p), ' sensors are calcuratiing with DG']);

[S, time] = F_DG(U(:, 1:r), p);
error = F_LinearEstimation(Xtrain, Ytrain, Xtest, Ytest, ...
    S, lambda2, 'NMSE');
errors('DG') = error;
times('DG') = time(end) + tSVD;

%% BDG ------------------------------------------------------------
disp([num2str(p), ' sensors are calcuratiing with BDG']);

[S, time] = F_BDG(U(:, 1:r), Sigma(1:r, 1:r), ...
    U(:, r+1:end), Sigma(r+1:end, r+1:end), p);
error = F_LinearEstimation(Xtrain, Ytrain, Xtest, Ytest, ...
    S, lambda2, 'NMSE');
errors('BDG') = error;
times('BDG') = time(end) + tSVD;

end

function errors = F_selection_lambda(Xtrain, Ytrain, Xtest, Ytest, ...
    p, lambdas, lambdas2)

L = length(lambdas);
errors = zeros(L, 1);

for i = 1:L
    lambda = lambdas(i);
    lambda2 = lambdas2(i);
    %% GREG -----------------------------------------------------------
    disp(['lambda = ', num2str(lambda)])
    disp([num2str(p), ' sensors are calcuratiing with EEG']);
    
    % [dS, time] = EEG(Xtrain, Ytrain, pmax, lambda);
    S = F_GREG(Xtrain, Ytrain, p, lambda);
    error = F_LinearEstimation(Xtrain, Ytrain, Xtest, Ytest, ...
        S, lambda2, 'NMSE');
    errors(i) = error;
end

end

function h = F_plot_mean(x, Y)

meanY = mean(Y, 2);
innan = ~isnan(meanY);
h = plot(x(innan), meanY(innan));

end

function h = F_errorfill_meanstd(x, Y)

meanY = mean(Y, 2);
stdY  = std(Y, 0, 2) / sqrt(size(Y, 2));
innan = ~isnan(meanY);
x2 = [x(innan); flipud(x(innan))];
y2 = [meanY(innan) - stdY(innan); flipud(meanY(innan) + stdY(innan))];
h = fill(x2, y2, 'r');

end

function F_change_size(f, width, height)

set(f,'PaperPositionMode','auto')
set(f, 'Units', 'centimeters')
pos    = get(gcf, 'Position');
pos(3) = width;
pos(4) = height;
set(f, 'Position', pos)
set(f, 'PaperSize', pos(3:4))

end
