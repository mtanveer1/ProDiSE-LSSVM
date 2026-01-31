function [alpha, b, P_indices, time] = ProDiSE_LSSVM_Train(X, Y, kerfpara, lambda, m,seed)

Ns_ratio=0.1; 
E=20; 
    fprintf('\n--- Starting ProDiSE-LSSVM Training (V4 Weighted) ---\n');
    
    tic
    [P_indices, ~] = ProDiSE(X, Y, m, E, Ns_ratio,seed);
    
    X_P = X(P_indices, :);
    Y_P = Y(P_indices);
    m_actual = size(X_P, 1);
    
    fprintf('Selected subset size (m): %d points.\n', m_actual);
    
    % --- LSSVM Dual System Setup and Solve ---
    K = kernelfun(X_P, kerfpara, X_P);
    Omega = K + (1/lambda) * eye(m_actual); %score_matrix;%
    A = [Omega, ones(m_actual, 1);
         ones(1, m_actual), 0];
    B = [Y_P; 0];
    
    try
        Solution = A \ B;
    catch
        error('Matrix singularity error. Try increasing lambda or reducing subset size m.');
    end
    
    alpha = Solution(1:m_actual);
    b = Solution(m_actual + 1);
    fprintf('LSSVM training complete. Parameters found.\n');
    time=toc;
end
%% 
function [P_indices, E_SelScore] = ProDiSE(X, Y, m, E, Ns_ratio,seed,Q)
% ProDiSE: Implements the Variance-Weighted linear combination.

    [N, D] = size(X);
    k = 1; % 1-NN distance
    classes = unique(Y);
    num_classes = length(classes);
    Ns = max(2, round(N * Ns_ratio)); 
    eps = 1e-6; % Small constant to prevent division by zero for variance

    SelScore_matrix = zeros(N, E);
    
    for e = 1:E
        rng(seed(e)); 
        subsample_indices = randperm(N, Ns);
        X_sub = X(subsample_indices, :);
        Y_sub = Y(subsample_indices);
        
        SelScore_e = zeros(N, 1);
        
        for c_idx = 1:num_classes
            current_class = classes(c_idx);
            global_class_indices = find(Y == current_class);
            sub_class_indices = find(Y_sub == current_class);
            
            if length(sub_class_indices) < 2
                continue; 
            end
            
            X_sub_C = X_sub(sub_class_indices, :);
            
            % --- 1. Calculate Raw Scores (LDS and GCS) ---
            D_allC_to_subC = pdist2(X(global_class_indices, :), X_sub_C, 'euclidean');
            D_sorted_C = sort(D_allC_to_subC, 2, 'ascend');
            LDS_C = D_sorted_C(:, k); % Raw Local Distance Score (r)
            
            X_centroid_sub_C = mean(X_sub_C, 1);
            GCS_C = pdist2(X(global_class_indices, :), X_centroid_sub_C, 'euclidean'); % Raw Global Distance Score (g)
            
            % --- 2. Calculate Variances and Alpha ---
            sigma_LDS_sq = var(LDS_C);
            sigma_GCS_sq = var(GCS_C);
            
            % Variance-based weighting (Alpha_auto)
            denominator = sigma_LDS_sq + sigma_GCS_sq + eps;
            alpha_auto_C = sigma_LDS_sq / denominator;
            one_minus_alpha = sigma_GCS_sq / denominator; % (1 - alpha_auto)
            
            % --- 3. Normalization (Per-Class) ---
            LDS_norm_C = (LDS_C - min(LDS_C)) / (max(LDS_C) - min(LDS_C) + eps);
            GCS_norm_C = (GCS_C - min(GCS_C)) / (max(GCS_C) - min(GCS_C) + eps);
            
            % --- 4. Combination: Variance-Weighted Linear Sum ---
            SelScore_e(global_class_indices) = (alpha_auto_C .* LDS_norm_C) + (one_minus_alpha .* GCS_norm_C);
        end
        
        SelScore_matrix(:, e) = SelScore_e;
    end

    % 5. Ensemble Aggregation (75th percentile)
    q=0.75;
    E_SelScore = quantile(SelScore_matrix', q)';

    % 6. Final Selection
    P_indices = select_guaranteed_balanced_subset(Y, E_SelScore, m);
   
end
% --- Helper Function for Final Proportional Selection (Same as V3) ---
function P_indices = select_guaranteed_balanced_subset(Y, Scores, m)
    classes = unique(Y);
    l = length(Y);
    P_indices = [];
    m_sum = 0;
    
    for c = 1:length(classes)
        current_class = classes(c);
        class_indices = find(Y == current_class);
        l_c = length(class_indices);
        
        m_c = max(1, round(m * (l_c / l)));
        m_c = min(m_c, l_c); 
        m_sum = m_sum + m_c;
        
        class_scores = Scores(class_indices);
        [~, sorted_local_indices] = sort(class_scores, 'ascend');
        
        selected_local_indices = sorted_local_indices(1:m_c);
        P_indices_c = class_indices(selected_local_indices);
        
        P_indices = [P_indices; P_indices_c];
    end
    
    if length(P_indices) > m 
        current_scores = Scores(P_indices);
        [~, final_sort_idx] = sort(current_scores, 'ascend');
        P_indices = P_indices(final_sort_idx(1:m));
        
    elseif length(P_indices) < m 
        all_indices = (1:l)';
        unselected_indices = setdiff(all_indices, P_indices);
        
        remaining_scores = Scores(unselected_indices);
        [~, remaining_sort_idx] = sort(remaining_scores, 'ascend');
        
        needed = m - length(P_indices);
        P_indices = [P_indices; unselected_indices(remaining_sort_idx(1:needed))];
    end
end
