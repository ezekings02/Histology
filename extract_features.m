function features = extract_features(img, mask_cell, mask_nuclei)

    % Convert to HSV
    hsv = rgb2hsv(img);
    H = hsv(:,:,1); S = hsv(:,:,2); V = hsv(:,:,3);

    % Feature struct
    features = struct();

    %% RegionProps
    stats_cell = regionprops(mask_cell, 'Area', 'Perimeter', 'Solidity');
    stats_nuc = regionprops(mask_nuclei, 'Area', 'Perimeter', 'Solidity', 'Eccentricity', 'Extent');

    if ~isempty(stats_cell)
        features.Area_cell = stats_cell(1).Area;
        features.Perimeter_cell = stats_cell(1).Perimeter;
        features.Solidity_cell = stats_cell(1).Solidity;
    else
        features.Area_cell = 0; features.Perimeter_cell = 0; features.Solidity_cell = 0;
    end

    if ~isempty(stats_nuc)
        features.Area_nucleus = stats_nuc(1).Area;
        features.Perimeter_nucleus = stats_nuc(1).Perimeter;
        features.Solidity_nucleus = stats_nuc(1).Solidity;
        features.Eccentricity_nucleus = stats_nuc(1).Eccentricity;
        features.Extent_nucleus = stats_nuc(1).Extent;
        features.Circularity_nucleus = 4 * pi * stats_nuc(1).Area / (stats_nuc(1).Perimeter^2 + eps);
    else
        features = rmfield(features, fieldnames(features));  % Return empty if no nucleus
        return;
    end

    %% Color Features - mean + std of HSV inside masks
    features.Mean_Hue_cell = mean(H(mask_cell));
    features.Mean_Saturation_cell = mean(S(mask_cell));
    features.Mean_Value_cell = mean(V(mask_cell));
    features.Std_Value_cell = std(V(mask_cell));

    features.Mean_Hue_nucleus = mean(H(mask_nuclei));
    features.Std_Saturation_nucleus = std(S(mask_nuclei));

end