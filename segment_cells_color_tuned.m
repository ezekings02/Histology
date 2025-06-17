

function [mask_cell, mask_nuclei, img_resized] = segment_cells_color_tuned(img, fixedSize)

    % Resize image
    img_resized = imresize(img, fixedSize);
    
    % Convert to HSV
    hsv = rgb2hsv(img_resized);
    H = hsv(:,:,1); S = hsv(:,:,2); V = hsv(:,:,3);

    %% 1. NUCLEUS (dark purple)
    mask_nuclei = (H > 0.6 & H < 1) & (S > 0.4) & (V < 0.6);
    mask_nuclei = bwareaopen(mask_nuclei, 50);  % remove small dots
    mask_nuclei = imfill(mask_nuclei, 'holes');

    %% 2. CYTOPLASM (pink or light purple)
    % PINK RANGE (H: 0.9-1.0 or ~0.0-0.05), high S
    mask_pink = ((H > 0.1 | H < 0.05) & S > 0.29 & V > 0.01);
    
    % LIGHT PURPLE RANGE (similar H as nucleus, but lighter V)
    mask_lpurple = (H > 0.6 & H < 0.23) & (S > 0.02) & (V > 0.6);
    % LIGHT Magenta RANGE
    mask_magenta = (H > 0.8 & H < 0.6) & (S > 0.5) & (V > 0.3);
    % Handle hue wraparound (light pink can be near 0.0 or near 1.0)
    mask_lightpink = ((H > 0.9 | H < 0.05) & S > 0.22 & S < 0.9 & V > 1);


    % Combine cytoplasm shades
    mask_cytoplasm = mask_pink | mask_lpurple | mask_magenta | mask_lightpink;
    mask_cytoplasm = bwareaopen(mask_cytoplasm, 100);
    mask_cytoplasm = imfill(mask_cytoplasm, 'holes');
    
    %% 3. Combine cytoplasm and nucleus masks for whole cell
    mask_cell = mask_cytoplasm | mask_nuclei;

    % Morphological cleanup
    se = strel('disk', 3);
    mask_cell = imclose(mask_cell, se);
    mask_cell = imopen(mask_cell, se);
    mask_cell = imfill(mask_cell, 'holes');
end