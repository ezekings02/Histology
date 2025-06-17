% Set up folders
baseDir = 'bloodcells_4classes_500';
categories = {'basophils', 'eosinophils', 'erythroblasts', 'immunoglobulins'};

% Parameters
fixedSize = [256, 256];
numSamples = 4;
imgs = cell(1, numSamples);
cells = cell(1, numSamples);
nuclei = cell(1, numSamples);

% Collect all image paths
imgPaths = {};
for i = 1:length(categories)
    folder = fullfile(baseDir, categories{i});
    files = dir(fullfile(folder, '*.jpg')); 
    for j = 1:length(files)
        imgPaths{end+1} = fullfile(folder, files(j).name);
    end
end

% Randomly pick 4 images to display
rng(50); % for reproducibility
randIdx = randperm(length(imgPaths), numSamples);
selectedPaths = imgPaths(randIdx);

% Process selected images
for k = 1:numSamples
    img = imread(selectedPaths{k});
    [mask_cell, mask_nuclei, resized_img] = segment_cells_color_tuned(img, fixedSize);
    imgs{k} = resized_img;
    cells{k} = mask_cell;
    nuclei{k} = mask_nuclei;
end

% Display examples
figure;
for i = 1:numSamples
    subplot(3, numSamples, i), imshow(imgs{i}), title('Original');
    subplot(3, numSamples, i + numSamples), imshow(cells{i}), title('Cell Mask');
    subplot(3, numSamples, i + 2*numSamples), imshow(nuclei{i}), title('Nucleus Mask');
end

% Output base (optional for saving masks)
outputBase = 'segmentation_results';

% Initialize feature storage
allFeatures = [];
labels = {};
sampleCount = 0;
skippedImages = {};

% Process and extract features from all images
for i = 1:length(categories)
    cat = categories{i};
    inputFolder = fullfile(baseDir, cat);
    outputCellFolder = fullfile(outputBase, cat, 'cell');
    outputNucleiFolder = fullfile(outputBase, cat, 'nuclei');

    % Create output folders if needed
    if ~exist(outputCellFolder, 'dir'), mkdir(outputCellFolder); end
    if ~exist(outputNucleiFolder, 'dir'), mkdir(outputNucleiFolder); end

    files = dir(fullfile(inputFolder, '*.jpg'));

    for j = 1:length(files)
        imgPath = fullfile(inputFolder, files(j).name);
        try
            img = imread(imgPath);
            [mask_cell, mask_nuclei, resized_img] = segment_cells_color_tuned(img, fixedSize);
        catch ME
            warning("Segmentation failed for image: %s. Reason: %s", imgPath, ME.message);
            skippedImages{end+1} = sprintf("Segmentation failed: %s", imgPath);
            continue;
        end

        % Save masks
        [~, name, ~] = fileparts(files(j).name);
        imwrite(mask_cell, fullfile(outputCellFolder, ['mask_cell_' name '.png']));
        imwrite(mask_nuclei, fullfile(outputNucleiFolder, ['mask_nuclei_' name '.png']));

        try
            f = extract_features(resized_img, mask_cell, mask_nuclei);
            if isempty(f)
                warning("Empty features for image: %s", imgPath);
                skippedImages{end+1} = sprintf("Empty features: %s", imgPath);
                continue;
            end
        catch ME
            warning("Feature extraction failed for image: %s. Reason: %s", imgPath, ME.message);
            skippedImages{end+1} = sprintf("Feature extraction failed: %s", imgPath);
            continue;
        end

        allFeatures = [allFeatures; struct2array(f)];
        labels{end+1} = cat;
        sampleCount = sampleCount + 1;
    end
end

% Create feature table
if isempty(allFeatures)
    error('No features extracted. Check if segmentation or feature extraction is failing for all images.');
end

featureNames = fieldnames(f);
T = array2table(allFeatures, 'VariableNames', featureNames);

% Ensure the number of labels matches the number of rows
if height(T) ~= numel(labels)
    warning("Mismatch between feature rows (%d) and labels (%d). Truncating labels...", height(T), numel(labels));
    labels = labels(1:height(T)); % truncate if needed
end

T.Label = categorical(labels(:)); % add labels

% Save features for later use
writetable(T, 'features_extracted.csv');
disp("✅ Feature extraction completed. Total samples: " + string(sampleCount));

% Show skipped images
if ~isempty(skippedImages)
    fprintf("\n⚠️ Skipped %d images due to errors:\n", length(skippedImages));
    for i = 1:length(skippedImages)
        fprintf("  - %s\n", skippedImages{i});
    end
else
    fprintf("\n✅ No images were skipped.\n");
end

%% -----------------------------
% Prepare data
%% -----------------------------
Y = T.Label;
X = normalize(T{:,1:end-1});  % Exclude label column

% Split into train/test sets
rng('default'); % for reproducibility
cv = cvpartition(Y, 'HoldOut', 0.3);
Xtrain = X(training(cv), :);
Ytrain = Y(training(cv));
Xtest = X(test(cv), :);
Ytest = Y(test(cv));

%% -----------------------------
% SVM Classifier on All Features
%% -----------------------------
fprintf("🔁 Training SVM on all features...\n");
tic;
svmAll = fitcecoc(Xtrain, Ytrain);
timeAll = toc;

YpredAll = predict(svmAll, Xtest);
accAll = sum(YpredAll == Ytest) / numel(Ytest) * 100;
fprintf("✅ SVM Accuracy (All Features): %.2f%% | Training Time: %.2f sec\n", accAll, timeAll);

%% -----------------------------
% Optimized PCA Search
%% -----------------------------
fprintf("\n🔍 Searching for minimal PCA components with ≥ %.2f%% accuracy...\n", accAll - 1);
[coeff, ~, ~, ~, explained] = pca(Xtrain);

maxComp = size(Xtrain, 2);
accList = zeros(maxComp,1);
threshold = accAll - 1;
bestNum = 0;
bestAcc = 0;

for k = 1:maxComp
    XtrainK = Xtrain * coeff(:, 1:k);
    XtestK  = (Xtest - mean(Xtrain)) * coeff(:, 1:k);

    model = fitcecoc(XtrainK, Ytrain);
    preds = predict(model, XtestK);
    acc = mean(categorical(preds) == categorical(Ytest)) * 100;
    accList(k) = acc;

    if acc >= threshold && bestNum == 0
        bestNum = k;
        bestAcc = acc;
    end
end

fprintf("✅ Minimum components for ≥ %.2f%% accuracy: %d components (Accuracy = %.2f%%)\n", threshold, bestNum, bestAcc);

% Plot accuracy vs components
figure;
plot(1:maxComp, accList, '-o');
xlabel('Number of PCA Components');
ylabel('Accuracy (%)');
title('Accuracy vs. PCA Components');
grid on;

% -----------------------------
% Assess Performance for Eosinophils
% -----------------------------
% -----------------------------
% Confusion Matrix Analysis
% -----------------------------
% Generate confusion matrix
[confMat, order] = confusionmat(Ytest, YpredAll);

% Convert order to cell array if it's categorical
if iscategorical(order)
    order = cellstr(order);
end

% Display the confusion matrix
figure;
confusionchart(confMat, order);
title('Confusion Matrix - All Features');

% Print detailed metrics for each class
fprintf('\n📊 Detailed Performance Metrics:\n');
fprintf('%-15s %-10s %-10s %-10s %-10s %-10s\n', ...
    'Class', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'Support');

for i = 1:length(order)
    class = order{i};
    true_positives = confMat(i,i);
    false_positives = sum(confMat(:,i)) - true_positives;
    false_negatives = sum(confMat(i,:)) - true_positives;
    true_negatives = sum(confMat(:)) - true_positives - false_positives - false_negatives;
    support = sum(confMat(i,:));
    
    % Calculate metrics with checks for division by zero
    precision = safe_divide(true_positives, (true_positives + false_positives));
    recall = safe_divide(true_positives, (true_positives + false_negatives));
    specificity = safe_divide(true_negatives, (true_negatives + false_positives));
    f1 = safe_divide(2 * (precision * recall), (precision + recall));
    
    % Print formatted results
    fprintf('%-15s %-10.2f %-10.2f %-10.2f %-10.2f %-10d\n', ...
        class, precision, recall, specificity, f1, support);
end

% Calculate and print overall metrics
overall_accuracy = sum(diag(confMat)) / sum(confMat(:));
fprintf('\n✅ Overall Accuracy: %.2f%%\n', overall_accuracy * 100);

% Print confusion matrix to command window
fprintf('\nConfusion Matrix (Counts):\n');
fprintf('%-15s', 'True\Pred');
fprintf('%-10s', order{:});
fprintf('\n');

for i = 1:length(order)
    fprintf('%-15s', order{i});
    fprintf('%-10d', confMat(i,:));
    fprintf('\n');
end

% Helper function to prevent division by zero
function result = safe_divide(numerator, denominator)
    if denominator == 0
        result = 0;
    else
        result = numerator / denominator;
    end
end