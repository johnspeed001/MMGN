function draw_sampling_tasks_grid
    % 主函数
    rng(0);                % 固定随机种子
    n         = 10;        % 10x10 矩阵
    numTasks  = 4;
    numTimes  = 5;

    baseField = makeBaseField(n);  % 平滑背景场

    figure('Color','w','Position',[100 100 900 700]);

    % 预先保存每个 subplot 的句柄，后面画 Task 标签和时间轴会用到
    axMat = gobjects(numTasks, numTimes);

    %---------------- Task 1: fixed number + fixed locations ----------------
    N1 = 20;
    mask_fixed = sampleCells(n, N1);  % 固定位置

    for t = 1:numTimes
        idx         = (1-1)*numTimes + t;
        axMat(1, t) = subplot(numTasks, numTimes, idx);
        drawGrid(axMat(1, t), baseField, mask_fixed);
    end

    %---------------- Task 2: random number + fixed locations ----------------
    poolSize = 60;
    total    = n * n;
    poolIdx  = randperm(total, poolSize);   % 固定一批候选格子

    for t = 1:numTimes
        Nt  = randi([15, 50]);              % 每个时间步不同数量
        Nt  = min(Nt, poolSize);
        idx_sel = poolIdx(randperm(poolSize, Nt));

        mask = false(n, n);
        [r, c] = ind2sub([n, n], idx_sel);
        mask(sub2ind([n, n], r, c)) = true;

        idx         = (2-1)*numTimes + t;
        axMat(2, t) = subplot(numTasks, numTimes, idx);
        drawGrid(axMat(2, t), baseField, mask);
    end

    %---------------- Task 3: fixed number + random locations ----------------
    N3 = 20;
    for t = 1:numTimes
        mask = sampleCells(n, N3);          % 每次位置随机

        idx         = (3-1)*numTimes + t;
        axMat(3, t) = subplot(numTasks, numTimes, idx);
        drawGrid(axMat(3, t), baseField, mask);
    end

    %---------------- Task 4: random number + random locations ----------------
    for t = 1:numTimes
        N4   = randi([10, 60]);
        mask = sampleCells(n, N4);

        idx         = (4-1)*numTimes + t;
        axMat(4, t) = subplot(numTasks, numTimes, idx);
        drawGrid(axMat(4, t), baseField, mask);
    end

    % 统一把子图整体往上抬一点，给下面时间轴留空间
    set(gcf, 'Units', 'normalized');
    for i = 1:numTasks
        for t = 1:numTimes
            pos = get(axMat(i,t),'Position');
            pos(2) = pos(2) + 0.03;   % 上移一点
            set(axMat(i,t),'Position',pos);
        end
    end

    %---------------- 每行上方画 Task 标签（水平） ----------------
    taskLabels = {'Task 1', 'Task 2', 'Task 3', 'Task 4'};
    addTaskLabelsAboveRows(axMat, taskLabels);

    %---------------- 最下面画一根时间轴，并在中间写 t1~t5 ----------------
    addBottomTimeAxis(axMat(end, :), numTimes);
end


%% ========= 辅助函数们 =========

function field = makeBaseField(n)
    % 构造平滑变化的 2D 场，做背景渐变
    [X, Y] = meshgrid(linspace(0, 1, n), linspace(0, 1, n));
    field  = (X + Y) / 2;           % 左下到右上渐变
    field  = field / max(field(:)); % 归一化到 [0,1]
end


function mask = sampleCells(n, k)
    % 在 n×n 网格随机选 k 个格子，返回逻辑 mask
    total = n * n;
    k     = min(k, total);
    idx   = randperm(total, k);

    mask  = false(n, n);
    mask(idx) = true;
end


function drawGrid(ax, baseField, mask)
    % 在指定坐标轴 ax 上画 10x10 彩色网格
    axes(ax);
    n  = size(baseField, 1);

    % 背景渐变；选中格子统一为 colormap 的高亮色
    bg   = baseField / max(baseField(:)) * 0.6;  % 背景 0~0.6
    data = bg;
    data(mask) = 1.0;                            % 被选中格子 = 1，统一颜色

    imagesc(data, [0 1]);
    axis image ij;
    set(ax, 'XTick', [], 'YTick', [], 'Box', 'on');
    colormap(ax, parula);   % 可改为 turbo / jet 等

    hold on;
    % 画网格线
    for k = 0.5:1:n+0.5
        plot([0.5, n+0.5], [k, k], 'k-', 'LineWidth', 0.3);
        plot([k, k], [0.5, n+0.5], 'k-', 'LineWidth', 0.3);
    end
    hold off;
end


function addTaskLabelsAboveRows(axMat, taskLabels)
    % 在每一行第一个子图上方，画水平的 Task 标签
    numTasks = size(axMat, 1);
    set(gcf,'Units','normalized');

    for i = 1:numTasks
        axFirst = axMat(i,1);
        pos = get(axFirst, 'Position');   % [x y w h] 归一化坐标

        % 子图水平中心
        xCenter = pos(1) + pos(3)/2;
        % 以子图上边为基准，稍微往上 0.005（可以自己再微调）
        yCenter = pos(2) + pos(4) + 0.015;

        boxW = 0.10;
        boxH = 0.03;

        annotation('textbox', ...
            [xCenter - boxW/2, yCenter - boxH/2, boxW, boxH], ...
            'String', taskLabels{i}, ...
            'EdgeColor', 'none', ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', ...
            'FontSize', 12);
    end
end



function addBottomTimeAxis(axRow, numTimes)
    % 只在最下面一行画一根时间轴，并把 t1~t5 写在每个小图下方的“中间”
    set(gcf,'Units','normalized');

    % 这一行最左和最右 subplot 的位置
    posFirst = get(axRow(1),   'Position');
    posLast  = get(axRow(end), 'Position');

    xStart = posFirst(1);
    xEnd   = posLast(1) + posLast(3);

    % 时间轴放在所有 subplot 下方一点
    yAxis  = posFirst(2) - 0.05;

    % 画箭头
    annotation('arrow', [xStart, xEnd], [yAxis, yAxis], ...
               'LineWidth', 1.2);

    % 在箭头下方，把 t1~t5 放到对应列 subplot 的“水平中心”
    for t = 1:numTimes
        posCol  = get(axRow(t),'Position');
        xCenter = posCol(1) + posCol(3)/2;   % 这一列小图的中心
        yText   = yAxis - 0.035;             % 略低于时间轴

        annotation('textbox', [xCenter-0.02, yText, 0.04, 0.03], ...
                   'String', sprintf('t%d', t), ...
                   'EdgeColor', 'none', ...
                   'HorizontalAlignment', 'center', ...
                   'VerticalAlignment', 'middle', ...
                   'FontSize', 10);
    end
end
