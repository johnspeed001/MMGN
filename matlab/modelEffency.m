%% model_efficiency_plot.m
clear; clc; close all;

% ================== 1. 从 CSV 里读数据 ==================
projectRoot = fileparts(fileparts(mfilename('fullpath')));
inputFile = fullfile(projectRoot, 'data', 'benchmarks', 'model_eval.csv');
outputDir = fullfile(projectRoot, 'outputs', 'matlab');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end
T = readtable(inputFile, 'TextType','string');

% 想要画的 5 个模型在 CSV 中的名称（注意 FFN_P / FFN_G）
models_in_csv = ["M2GNet","ResMLP","FFN_P","FFN_G","SIREN"];

% 对应的显示名称（与原脚本一致）
modelNames = {'M2GNet','ResMLP','FFN+P','FFN+G','SIREN'};

% 按照 models_in_csv 的顺序，从表中取出行
[tf, loc] = ismember(models_in_csv, T.model);
if ~all(tf)
    error('model_eval.csv 缺少以下模型：%s', ...
        strjoin(cellstr(models_in_csv(~tf)), ', '));
end
T = T(loc, :);

% 横坐标：MSE（test_mse，越小越好）
mse = T.test_mse;                  % 列向量

% 纵坐标：单次推理时间 / s （CSV 里是 ms）
t_infer = T.infer_time_ms / 1000;  % 转成秒

% 可训练参数个数（param_M 是 M 个参数）
nParams = T.param_M * 1e6;         % M -> 绝对参数个数

% 检查长度一致
assert(numel(modelNames)==numel(mse) && ...
       numel(mse)==numel(t_infer) && ...
       numel(t_infer)==numel(nParams), ...
       '数据长度不一致，请检查！');

% ================== 2. 把参数量映射成圆的大小 ==================
% scatter 的 size 单位是 “点^2”，这里做一个简单线性映射
minSize = 400;   % 最小气泡大小
maxSize = 1600;  % 最大气泡大小
pmin = min(nParams);
pmax = max(nParams);
S = minSize + (nParams - pmin) ./ max(pmax - pmin, eps) * (maxSize - minSize);

% ================== 3. 绘图 ==================
figure('Color','w'); hold on; box on;

% 自定义每个模型的颜色（与原脚本对应）
colors = [...
    1.0 0.6 0.6;   % M2GNet
    0.6 0.7 1.0;   % ResMLP
    0.6 0.9 0.6;   % FFN+P
    0.9 0.9 0.6;   % FFN+G
    0.6 0.9 0.9];  % SIREN

for i = 1:numel(modelNames)
    scatter(mse(i), t_infer(i), S(i), ...
        'MarkerFaceColor', colors(i,:), ...
        'MarkerEdgeColor', 'k', ...
        'LineWidth', 0.8);
end

% 给每个气泡加文字标签（模型名）
for i = 1:numel(modelNames)
    text(mse(i), t_infer(i), modelNames{i}, ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', ...
        'FontSize',10, ...
        'FontName','Times New Roman');
end

% 坐标轴
xlabel('MSE','FontName','Times New Roman','FontSize',12);
ylabel('Inference time per instance (sec)','FontName','Times New Roman','FontSize',12);

set(gca,'FontName','Times New Roman','FontSize',11);
grid on;

% ===== 根据数据自适应坐标范围（横轴右边多留一点给气泡图例） =====
xMin = min(mse); xMax = max(mse);
dx = xMax - xMin; if dx == 0, dx = 1; end
xlim([xMin - 0.05*dx, xMax + 0.45*dx]);   % 右侧多留 0.45*dx 的空间

yMin = min(t_infer); yMax = max(t_infer);
dy = yMax - yMin; if dy == 0, dy = 1e-3; end
ylim([yMin - 0.15*dy, yMax + 0.55*dy]);   % 上方多留空间给灰色气泡和文字

% 把主坐标轴挤一点，为右侧“输入参数区域”留空间
ax = gca;
ax.Position = [0.11 0.12 0.55 0.78];  % [left bottom width height]

% ================== 4. 在图中上方做“可训练参数”图例 ==================
% 图例中想显示的两个参考参数量
paramLegend = [5e5, 1e6];  % 例如：500K 和 1.00M
S_legend = minSize + (paramLegend - pmin) ./ max(pmax - pmin, eps) * (maxSize - minSize);

% 气泡图例位置（用数据坐标表示）
x0 = xMin + 0.70*dx;          % 第一个灰色气泡的 x
x1 = xMin + 0.82*dx;          % 第二个灰色气泡的 x
y0 = yMax + 0.25*dy;          % 两个气泡的 y（在数据点上方）

% 画两个灰色气泡
scatter(x0, y0,   S_legend(1), 'MarkerFaceColor',[0.7 0.7 0.7], 'MarkerEdgeColor','k');
scatter(x1, y0,   S_legend(2), 'MarkerFaceColor',[0.7 0.7 0.7], 'MarkerEdgeColor','k');

% “Trainable params” 放在两颗灰色气泡正上方
xMid = (x0 + x1) / 2;
text(xMid, y0 + 0.08*dy, 'Trainable params', ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','bottom', ...
    'FontName','Times New Roman','FontSize',10);

% 标注两个参考数值
text(x0,   y0-0.05*dy, '500K', ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','top', ...
    'FontName','Times New Roman','FontSize',9);
text(x1,   y0-0.05*dy, '1.00M', ...
    'HorizontalAlignment','center', ...
    'VerticalAlignment','top', ...
    'FontName','Times New Roman','FontSize',9);

% ================== 5. 右侧“输入参数”专用区域 ==================
paramText = {
    '\bfInput parameters:'
    'Grid: 128 \times 128'
    'Time steps: 50'
    'Sampling rate: 0.05'
    'Training instances: 1000'
    'Test instances: 200'
    };

annotation('textbox', [0.70 0.55 0.26 0.35], ...
    'String', paramText, ...
    'Interpreter','tex', ...
    'FontName','Times New Roman', ...
    'FontSize',9, ...
    'EdgeColor',[0.5 0.5 0.5], ...
    'LineWidth',0.8, ...
    'BackgroundColor','none');

% ================== 6. 高分辨率导出（可选） ==================
set(gcf,'PaperPositionMode','auto');
print(gcf,fullfile(outputDir,'model_efficiency'),'-dpng','-r600');
