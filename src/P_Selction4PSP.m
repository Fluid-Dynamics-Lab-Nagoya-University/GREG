%% Main program

clear
close all

%% Parameters =======================================================
lambdas = [0, 10 .^ (-10:1:0)];
pmax = 10;
r = 10;
ParentDir = '../data/resultsPSP';

%% Preparation of output directories ================================

%% Preprocces for PSP dataset ==================================
text = 'Readinng/Arranging a PSP dataset';
disp(text);
load ../data/PSP X Y

X = F_Preprocess(X); % surface pressure differece
N = size(X, 1);
M = size(X, 2);

% X(:, round((M + 1) / 2)) = [];
% Y(:, round((M + 1) / 2)) = [];
% M = M - 1;

for lambda = lambdas
    fprintf('lambda = %5.0e \n', lambda)
    mkdir(ParentDir);
    Dir = [ParentDir, '/lambda', num2str(lambda, '%5.0e')];
    mkdir(Dir);

    %% Performing for training set
    itrain = 1:M;
    itest  = 1:M;
        
    [Xtrain, Ytrain, Xtest, Ytest, U, Sigma, tSVD] = ...
        F_DataPreprocessing4Estimation(X, Y, itrain, itest);
    
    tbl = F_selection(Xtrain, Ytrain, Xtest, Ytest, U, Sigma, r, pmax, ...
        lambda, tSVD);
    
    save([Dir, '/results'], 'tbl')

    %% Performing for CV sets
    [Itrain, Itest] = F_SymmetricLOO(size(X, 2));
    N_CV = length(Itrain);
    for j = 1:N_CV
        itrain = Itrain{j};
        itest  = Itest {j};
        
        [Xtrain, Ytrain, Xtest, Ytest, U, Sigma, tSVD] = ...
            F_DataPreprocessing4Estimation(X, Y, itrain, itest);
        
        tbl = F_selection(Xtrain, Ytrain, Xtest, Ytest, U, Sigma, r, pmax, ...
            lambda, tSVD);
        
        save([Dir, '/resultsCV', num2str(j)], 'tbl', 'itest', 'itrain')
    end
end

%% Arragne results data
clear

ivec = 1:6;
lambdas = [0, 10 .^ (-10:1:0)];
N_CV = 25;
Dir = '../data/resultsPSP';

L = length(ivec);
Errors = cell(L, 1);
ErrorTrs = cell(L, 1);
NormKs = cell(L, 1);
Times = cell(L, 1);
Ps = cell(L, 1);
dSs = cell(L, 1);

for lambda = lambdas
    for k = 1:N_CV
        load([Dir, '/lambda', num2str(lambda, '%5.0e'), ...
            '/resultsCV', num2str(k)], 'tbl')
        tbls = tbl.table;
        tbls = tbls{ivec, :};
        for i = 1:L
            if k == 1
                p = tbls{i}.p;
                Ps{i} = p;

                Np = length(p);
                Errors  {i} = zeros(Np, N_CV);
                ErrorTrs{i} = zeros(Np, N_CV);
                Times   {i} = zeros(Np, N_CV);
                dSs     {i} = zeros(Np, N_CV);
            end
            Errors  {i}(:, k) = tbls{i}.error;
            ErrorTrs{i}(:, k) = tbls{i}.error_tr;
            NormKs  {i}(:, k) = tbls{i}.normK;
            Times   {i}(:, k) = tbls{i}.time;
            dSs     {i}(:, k) = tbls{i}.dS;
        end
    end

    tbl2 = table('Size', [L, 2], ...
        'VariableTypes', {'string', 'struct'}, ...
        'VariableNames', {'selection', 'results'});
    selection = tbl.selection;
    tbl2.selection = selection(ivec);
    for i = 1:L
        tbl2.results(i, :).p        = Ps{i};
        tbl2.results(i, :).error    = Errors{i};
        tbl2.results(i, :).error_tr = ErrorTrs{i};
        tbl2.results(i, :).normK    = NormKs{i};
        tbl2.results(i, :).time     = Times{i};
        tbl2.results(i, :).dS       = dSs{i};
    end
    save([Dir, '/lambda', num2str(lambda, '%5.0e'), ...
        '/resultsCV'], 'tbl2')
end

%% Plot selected sensors
clear

flagCV = 0;
ps = 10;
ivec = 1:6;
clims1    = [-2.5 0];
clims2    = [0, 0.75];
lambdas = 1e-4;

mkdir('../pngs')
mkdir('../pdfs')
OutputBase1 = '../pngs/PSP_sensor_';
OutputBase2 = '../pdfs/PSP_sensor_';

load ../data/PSP X Y mask_Ahmed mask_X
X = F_Preprocess(X);

XX = F_Preprocess2(X);
RMSE_XX = vecnorm(XX, 2, 2) / sqrt(size(XX, 2));

for lambda = lambdas
    ParentDir = '../data/resultsPSP';
    Dir = [ParentDir, '/lambda', num2str(lambda, '%5.0e')];
    if flagCV
        load([Dir, '/resultsCV2'], 'tbl')
    else
        load([Dir, '/results'], 'tbl')
    end
    names = tbl.selection;
    tbls = tbl.table;
    names = names(ivec);
    tbls = tbls{ivec, :};
    for i = 1:length(names)
        for p = ps
            dS = tbls{i}.dS;
            S = dS(1:p);
            figure
            F_imshow_AhmedUpper(RMSE_XX, clims2, mask_X, mask_Ahmed)
            hold on
            F_PlotSensors_Upper(RMSE_XX, p, S, mask_X);
            hold off
            FileName1 = [OutputBase1, names{i}, '_lambda', ...
                num2str(lambda, '%5.0e'), '_p', num2str(p), '.png'];
            FileName2 = [OutputBase2, names{i}, '_lambda', ...
                num2str(lambda, '%5.0e'), '_p', num2str(p), '.pdf'];
            delete(FileName1)
            exportgraphics(gca, FileName1)
            delete(FileName2)
            exportgraphics(gca, FileName2)
        end
    end
end

%% Plot CV results
clear

blue = [0; 83; 168] / 255;
red  = [192; 0; 0] / 255;
green = [0; 110; 79] / 255;
purple = [102; 0; 102] / 255;
yellow = [246; 191; 0] / 255;
grey = 0.2 * [1; 1; 1];
Color = {grey, blue, red, green, purple, yellow};

white = [1; 1; 1];
alpha = 0.25;

width = 12;
height = 11;
FontSize = 13;
FontName = 'Times New Roman';
LineWidth = 1;

ivec = 1:6;
lambdas = 1e-4;
InputParentDir = '../data/resultsPSP';
OutputBase1 = '../pngs/PSP_';
OutputBase2 = '../pdfs/PSP_';

flag = 0;

for lambda = lambdas
    Dir = [InputParentDir, '/lambda', num2str(lambda, '%5.0e')];
    load([Dir, '/resultsCV'], 'tbl2')
    names = tbl2.selection;
    names(1) = "GREG $$ (\tilde{\lambda} = 0) $$";
    names(2) = "GREG $$ (\tilde{\lambda} = 10^{-4}) $$";
    results = tbl2.results;
    names = names(ivec);
    results = results(ivec);

    H = gobjects(1, length(names));

    figure
    hold on
    for i = 1:length(names)
        result = results(i);
        p = result.p;
        Y = result.error;
        h = F_errorfill_meanstd(p, Y);
        h.FaceColor = Color{i};
        h.FaceAlpha = alpha;
        h.EdgeColor = 'none';
    end
    for i = 1:length(names)
        result = results(i);
        p = result.p;
        error = result.error;
        h = F_plot_mean(p, error);
        h.Color = Color{i};
        h.LineWidth = LineWidth;
        h.DisplayName = names{i};
        H(i) = h;
    end
    lgd = legend(H);
    lgd.NumColumns = 2;
    lgd.Interpreter = 'Latex';
    xlabel('Number of Sensors, $$ p $$', 'Interpreter', 'Latex')
    ylabel('Estimation Error [deg]', 'Interpreter', 'Latex')
    grid on
    ax = gca;
    ax.Box = 'on';
    ax.FontSize = FontSize;
    ax.FontName = FontName;
    F_change_size(gcf, width, height)
    hold off
    FileName1 = [OutputBase1, 'CV_error_meanstd_lambda', ...
        num2str(lambda, '%5.0e'), '.png'];
    FileName2 = [OutputBase2, 'CV_error_meanstd_lambda', ...
        num2str(lambda, '%5.0e'), '.pdf'];
    delete(FileName1)
    exportgraphics(gca, FileName1)
    delete(FileName2)
    exportgraphics(gca, FileName2)

    figure
    hold on
    for i = 1:length(names)
        result = results(i);
        p = result.p;
        Y = result.error_tr;
        h = F_errorfill_meanstd(p, Y);
        h.FaceColor = Color{i};
        h.FaceAlpha = alpha;
        h.EdgeColor = 'none';
    end
    for i = 1:length(names)
        result = results(i);
        p = result.p;
        error = result.error_tr;
        h = F_plot_mean(p, error);
        h.Color = Color{i};
        h.LineWidth = LineWidth;
        h.DisplayName = names{i};
        H(i) = h;
    end
    lgd = legend(H);
    lgd.NumColumns = 2;
    lgd.Interpreter = 'Latex';
    xlabel('Number of Sensors, $$ p $$', 'Interpreter', 'Latex')
    ylabel('Estimation Error [deg]', 'Interpreter', 'Latex')
    grid on
    ax = gca;
    ax.Box = 'on';
    ax.FontSize = FontSize;
    ax.FontName = FontName;
    F_change_size(gcf, width, height)
    hold off
    FileName1 = [OutputBase1, 'CV_error_tr_meanstd_lambda', ...
        num2str(lambda, '%5.0e'), '.png'];
    FileName2 = [OutputBase2, 'CV_error_tr_meanstd_lambda', ...
        num2str(lambda, '%5.0e'), '.pdf'];
    delete(FileName1)
    exportgraphics(gca, FileName1)
    delete(FileName2)
    exportgraphics(gca, FileName2)

    figure
    hold on
    for i = 1:length(names)
        result = results(i);
        p = result.p;
        Y = result.normK;
        h = F_errorfill_meanstd(p, Y);
        h.FaceColor = Color{i};
        h.FaceAlpha = alpha;
        h.EdgeColor = 'none';
    end
    for i = 1:length(names)
        result = results(i);
        p = result.p;
        Y = result.normK;
        h = F_plot_mean(p, Y);
        h.Color = Color{i};
        h.LineWidth = LineWidth;
        h.DisplayName = names{i};
        H(i) = h;
    end
    lgd = legend(H);
    lgd.NumColumns = 2;
    lgd.Interpreter = 'Latex';
    xlabel('Number of Sensors, $$ p $$', 'Interpreter', 'Latex')
    ylabel('Norm of Gain, $$ \| K \|_{\rm F} $$', ...
        'Interpreter', 'Latex')
    grid on
    ax = gca;
    ax.Box = 'on';
    ax.FontSize = FontSize;
    ax.FontName = FontName;
    F_change_size(gcf, width, height)
%     legend(names)
    hold off
    FileName1 = [OutputBase1, 'CV_normK_meanstd_lambda', ...
        num2str(lambda, '%5.0e'), '.png'];
    FileName2 = [OutputBase2, 'CV_normK_meanstd_lambda', ...
        num2str(lambda, '%5.0e'), '.pdf'];
    delete(FileName1)
    exportgraphics(gca, FileName1)
    delete(FileName2)
    exportgraphics(gca, FileName2)

end

%% 

function tbls = F_selection(Xtrain, Ytrain, Xtest, Ytest, U, Sigma, r, pmax, ...
    lambda, tSVD)

ps = (1:pmax)';

tbls = table('Size', [0, 2], ...
    'VariableTypes', {'string', 'table'}, ...
    'VariableNames', {'selection', 'table'});

tbl = table('Size', [pmax, 6], ...
    'VariableTypes', {'int64', 'int64', 'double', 'double',...
    'double', 'double'}, ...
    'VariableNames', {'p', 'dS', 'error', 'error_tr', ...
    'normK', 'time'});
tbl.p = ps;

IlargeU = find(min(abs(U), [], 2) > 1e-10);
U = U(IlargeU, :);

%% GREG0 -----------------------------------------------------------
disp(['1-', num2str(pmax), ' sensors are calcuratiing with GREG0']);

[dS, time] = F_GREG(Xtrain, Ytrain, pmax, 0);

tbl2 = F_IncrementSet2tbl(dS, time, Xtrain, Ytrain, Xtest, Ytest, ...
    lambda, 0, tbl);

tbls = [tbls; {'GREG0', tbl2}];

%% GREG1 -----------------------------------------------------------
disp(['1-', num2str(pmax), ' sensors are calcuratiing with GREG1']);

[dS, time] = F_GREG(Xtrain, Ytrain, pmax, lambda);

tbl2 = F_IncrementSet2tbl(dS, time, Xtrain, Ytrain, Xtest, Ytest, ...
    lambda, 0, tbl);

tbls = [tbls; {'GREG1', tbl2}];


%% REG -----------------------------------------------------------
disp(['1-', num2str(pmax), ' sensors are calcuratiing with REG']);

[dS, time] = F_REG(Xtrain, pmax);

tbl2 = F_IncrementSet2tbl(dS, time, Xtrain, Ytrain, Xtest, Ytest, ...
    lambda, 0, tbl);

tbls = [tbls; {'REG', tbl2}];

%% OMP -----------------------------------------------------------
disp(['1-', num2str(pmax), ' sensors are calcuratiing with SOMP']);

[dS, time] = F_SOMP(Xtrain, Ytrain, pmax);

tbl2 = F_IncrementSet2tbl(dS, time, Xtrain, Ytrain, Xtest, Ytest, ...
    lambda, tSVD, tbl);

tbls = [tbls; {'SOMP', tbl2}];

%% QD ------------------------------------------------------------
disp(['1-', num2str(pmax), ' sensors are calcuratiing with DG']);

[dS, time] = F_DG(U(:, 1:r), pmax);

dS = IlargeU(dS);

tbl2 = F_IncrementSet2tbl(dS, time, Xtrain, Ytrain, Xtest, Ytest, ...
    lambda, tSVD, tbl);

tbls = [tbls; {'DG', tbl2}];

%% BDG ------------------------------------------------------------
disp(['1-', num2str(pmax), ' sensors are calcuratiing with BDG']);

[dS, time] = F_BDG(U(:, 1:r), Sigma(1:r, 1:r), ...
    U(:, r+1:end), Sigma(r+1:end, r+1:end), pmax);

dS = IlargeU(dS);

tbl2 = F_IncrementSet2tbl(dS, time, Xtrain, Ytrain, Xtest, Ytest, ...
    lambda, tSVD, tbl);

tbls = [tbls; {'BDG', tbl2}];

end


function out_tbl = F_IncrementSet2tbl(dS, time, Xtrain, Ytrain, Xtest, Ytest, ...
    lambda, tSVD, tbl)

error = F_IncrementLinearEstimation4(Xtrain, Ytrain, Xtest, Ytest, dS, ...
    lambda, 'RMSE');
error_tr = F_IncrementLinearEstimation4(Xtrain, Ytrain, Xtrain, Ytrain, ...
    dS, lambda, 'RMSE');
eS = F_LinearEstimation(Xtrain, Ytrain, Xtest, Ytest, dS, lambda, 'RMSE');

pmax = length(dS);
normK = zeros(pmax, 1);
for p = 1:pmax
    S = dS(1:p);
    normK(p) = F_ComputeNormK(Xtrain, Ytrain, S, lambda);
end

out_tbl = tbl;
out_tbl.dS = dS;
out_tbl.time = time + tSVD;
out_tbl.error = error;
out_tbl.error_tr = error_tr;
out_tbl.normK = normK;

fprintf('error of incremental computation of error: %9.2e \n', ...
    error(pmax) - eS)

end

function [Xtrain, Ytrain, Xtest, Ytest, U, Sigma, tSVD] = ...
    F_DataPreprocessing4Estimation(X, Y, itrain, itest)

meanX = mean(X(:, itrain), 2);
Xtrain = X(:, itrain) - meanX;
tic;
[U, Sigma, ~] = svd(Xtrain, 'econ');
% [U, Sigma, ~] = svds(Xtrain, r);
tSVD = toc;
Ytrain = Y(:, itrain);
Xtest = X(:, itest) - meanX;
Ytest = Y(:, itest);

end

function F_imshow_AhmedUpper(x, clims, mask_X, mask_Ahmed)

DisSize = [32  154];
imgSize = size(mask_Ahmed);

temp1 = zeros(DisSize);
temp2 = temp1;

temp1(:) = x(1:length(x)/2);
temp2(:) = x(length(x)/2+1:length(x));
temp     = [flip(temp2,1); temp1];

for ii = 1:numel(temp)
    if temp(ii) == 999 
    elseif temp(ii) == -999
    else
        if temp(ii) <= clims(1) + diff(clims)/256
            temp(ii) = clims(1) + diff(clims)/256;
        elseif temp(ii) >= clims(2) - diff(clims)/256
            temp(ii) = clims(2) - diff(clims)/256;
        end
    end
end

Img             = zeros(imgSize);
Img(:,:)        = 999;
Img(mask_X)       = temp(:);
Img(mask_Ahmed) = -999;

img = Img(82:169,60:253);    
imshow(img,clims,'Border','tight','InitialMagnification',400)
colormap([0.5 0.5 0.5; jet; 1 1 1])

hold on
plot([1 193], [44.5 44.5], 'k--')
hold off
xlim([5, 189])
ylim([10, 44.5])

end

function F_PlotSensors_Upper(X, p, isensors, mask_X)

%% 
DisSize     = [32  154];
spatialSize = size(X,1)/2;
imgSize     = size(mask_X);

%% 
temp1 = zeros(DisSize);
temp1(:) = 1:spatialSize;
temp     = [flip(temp1,1); temp1];

sImg         = zeros(imgSize);
sImg(mask_X) = temp(:);
simg         = sImg(82:169,60:253); 
for ii = 1:p
    [sX, sY] = find(simg == isensors(ii));
    plot(sY,sX,'wx', 'MarkerFaceColor', 'none','MarkerSize', 12, 'LineWidth', 3); %
end

end

function X = F_Preprocess(X)

X = X - fliplr(X);

end

function X = F_Preprocess2(X)

X = [X; fliplr(X)];

end

function h = F_plot_mean(x, Y)

meanY = mean(Y, 2);
h = plot(x, meanY);

end

function h = F_errorfill_meanstd(x, Y)

meanY = mean(Y, 2);
stdY  = std(Y, 0, 2) / sqrt(size(Y, 2));
x2 = [x; flipud(x)];
y2 = [meanY - stdY; flipud(meanY + stdY)];
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