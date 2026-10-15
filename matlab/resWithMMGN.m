clear; clc;

%% ---- Global settings: English font, stable preview ----
set(groot,'defaultFigureRenderer','opengl');
set(groot,'defaultTextInterpreter','tex');
set(groot,'defaultAxesFontName',   'Times New Roman');
set(groot,'defaultTextFontName',   'Times New Roman');
set(groot,'defaultLegendFontName', 'Times New Roman');
set(groot,'defaultLegendInterpreter','none');

%% ---- Basic settings ----
rates = ["0.05","0.25","0.5"];                 % 3 sampling rates on x-axis

% CSV 里模型名：ResMLP, FFN_P, FFN_G, SIREN, M2GNet, MMGN
algos_csv    = ["ResMLP","FFN_P","FFN_G","SIREN","MMGN","M2GNet"];
% 图例顺序与之对应，M2GNet 在最右
algos_legend = {'ResMLP','FFN+P','FFN+G','SIREN','MMGN','M2GNet'};

PNG_DPI = 600;
projectRoot = fileparts(fileparts(mfilename('fullpath')));
inputDir = fullfile(projectRoot, 'data', 'benchmarks');
outputDir = fullfile(projectRoot, 'outputs', 'matlab', 'resAll');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% nice colors (order matches algos_csv / algos_legend)
colors5 = [ ...
    0.40 0.65 0.87;   % ResMLP  (blue)
    0.57 0.82 0.61;   % FFN+P   (green)
    0.99 0.85 0.55;   % FFN+G   (yellow)
    1.00 0.60 0.60;   % SIREN   (red)
    0.30 0.30 0.30;   % MMGN    (dark gray)
    0.75 0.70 0.90];  % M2GNet  (purple/blue)

fsA = 10;    % axes font size
fsL = 9;     % legend font size
lw  = 0.8;   % edge line width
bw  = 0.8;   % bar width

%% ---- 四个任务的 CSV 文件 ----
taskFiles   = { ...
    fullfile(inputDir, 'task1_results.csv'), ...
    fullfile(inputDir, 'task2_results.csv'), ...
    fullfile(inputDir, 'task3_results.csv'), ...
    fullfile(inputDir, 'task4_results.csv')};

% 对应输出文件名前缀
taskPrefixes = { ...
    'T1', ...
    'T2', ...
    'T3', ...
    'T4'};

% 在图下方显示的文字
taskLabels = { ...
    'Task 1', ...
    'Task 2', ...
    'Task 3', ...
    'Task 4'};

metricNames  = {'PSNR','SSIM','MSE'};
metricTitles = {'PSNR (dB)','SSIM','MSE'};

nTask   = numel(taskFiles);
nMetric = numel(metricNames);
nAlgo   = numel(algos_csv);
nRate   = numel(rates);

% 每个任务的三种 metric 数据 (nAlgo x nRate)
dat_PSNR = cell(nTask,1);
dat_SSIM = cell(nTask,1);
dat_MSE  = cell(nTask,1);

% 用来统一 y 轴
psnr_all = [];
ssim_all = [];
mse_all  = [];

%% ---- 从 CSV 读入数据（忽略 promotion 行，只保留 6 个模型）----
for t = 1:nTask
    T = readtable(taskFiles{t}, 'TextType','string');

    % 丢掉 promotion 行
    if any(T.model == "promotion")
        T(T.model == "promotion", :) = [];
    end

    % 只保留需要画的 6 个模型，并按 algos_csv 的顺序重排
    [tf, loc] = ismember(algos_csv, T.model);
    if ~all(tf)
        error('File %s 缺少以下模型: %s', ...
            taskFiles{t}, strjoin(algos_csv(~tf), ', '));
    end
    T = T(loc, :);

    % 三个 metric 分别取出 (nAlgo x nRate)
    for mi = 1:nMetric
        metric = metricNames{mi};
        mat = zeros(nAlgo, nRate);

        for ai = 1:nAlgo
            for ri = 1:nRate
                % e.g. PSNR_0.05 -> PSNR_0_05
                rateTag  = char(rates(ri));          % "0.05"
                rateTag  = strrep(rateTag, '.', '_');% "0_05"
                varName  = sprintf('%s_%s', metric, rateTag);

                mat(ai, ri) = T.(varName)(ai);
            end
        end

        switch metric
            case 'PSNR'
                dat_PSNR{t} = mat;
                psnr_all = [psnr_all; mat(:)];
            case 'SSIM'
                dat_SSIM{t} = mat;
                ssim_all = [ssim_all; mat(:)];
            case 'MSE'
                dat_MSE{t} = mat;
                mse_all  = [mse_all;  mat(:)];
        end
    end
end

%% ---- 统一 y-limits（跨四个任务）----
psnr_min  = min(psnr_all);  psnr_max  = max(psnr_all);
psnr_span = max(psnr_max - psnr_min, 1e-6);
yl_psnr   = [psnr_min - 0.05*psnr_span, psnr_max + 0.06*psnr_span];
yl_psnr   = [floor(10*yl_psnr(1))/10, ceil(10*yl_psnr(2))/10];

ssim_min  = min(ssim_all);  ssim_max  = max(ssim_all);
yl_ssim   = [max(0, ssim_min - 0.02), min(1, ssim_max + 0.02)];

mse_min   = min(mse_all);   mse_max   = max(mse_all);
yl_mse    = [mse_min/1.2, mse_max*1.2];

%% ---- 画出四个任务，每个任务一个 figure，包含三个子图 ----
for t = 1:nTask
    prefix = taskPrefixes{t};

    % 新建一个 figure，三个横向子图
    fig = figure('Color','w','Units','centimeters', ...
                 'Position',[2 2 35 12], 'PaperPositionMode','auto');

    % --- 子图 1：PSNR ---
    ax1 = subplot(1,3,1);
    draw_subplot(ax1, dat_PSNR{t}, rates, algos_legend, colors5, ...
        metricTitles{1}, 'lin', yl_psnr, ...
        fsA, fsL, lw, bw, false);  % false: 不在这里画 legend

    % --- 子图 2：SSIM ---
    ax2 = subplot(1,3,2);
    draw_subplot(ax2, dat_SSIM{t}, rates, algos_legend, colors5, ...
        metricTitles{2}, 'lin', yl_ssim, ...
        fsA, fsL, lw, bw, false);

    % --- 子图 3：MSE（放 legend） ---
    ax3 = subplot(1,3,3);
    draw_subplot(ax3, dat_MSE{t}, rates, algos_legend, colors5, ...
        metricTitles{3}, 'lin', yl_mse, ...
        fsA, fsL, lw, bw, true);   % true: 在这一个子图上画 legend

    % 在最下方写明 Task1 ~ Task4
    annotation(fig,'textbox', ...
        'Units','normalized', ...
        'Position',[0 0.01 1 0.05], ...   % 靠近底部、居中
        'String', taskLabels{t}, ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', ...
        'FontName','Times New Roman', ...
        'FontSize', 11, ...
        'EdgeColor','none');

    % 导出整张图
    outname = sprintf('%s_metrics', prefix);
    exportgraphics(fig, fullfile(outputDir, [outname,'.png']), ...
        'Resolution', PNG_DPI, ...
        'BackgroundColor','white');
end

%% ---- Helper: draw one subplot on given axes ----
function draw_subplot(ax, dat_algXrate, rates, algos, colors5, metric_title, ...
                      ymode, ylim_val, fsA, fsL, lw, bw, showLegend)

    % dat_algXrate: (nAlgo x nRate)
    dat_plot = dat_algXrate.';   % -> (nRate x nAlgo) for bar()

    axes(ax); hold(ax,'on');

    b = bar(ax, dat_plot, 'BarWidth', bw);

    for i = 1:numel(b)   % each algorithm
        b(i).FaceColor = colors5(i,:);
        b(i).EdgeColor = 'none';
        b(i).LineWidth = lw;
    end

    % x-axis: sampling rates
    set(ax,'XTick',1:numel(rates), ...
           'XTickLabel',rates, ...
           'FontSize',fsA, ...
           'FontName','Times New Roman');
    xlabel(ax,'');

    % y-axis: only ticks, no label; metric written as title
    ylabel(ax,'');
    title(ax, metric_title, 'FontName','Times New Roman', 'FontSize', 12);

    grid(ax,'on');
    box(ax,'on');
    ax.Layer = 'top';

    if strcmpi(ymode,'log')
        set(ax,'YScale','log');
    end
    ylim(ax, ylim_val);

    % 只在指定的子图上画 legend
    if showLegend
        lg = legend(ax, algos, ...
            'Location','northeast', ...
            'FontName','Times New Roman', ...
            'FontSize',fsL);
        lg.Box = 'off';
    end
end
