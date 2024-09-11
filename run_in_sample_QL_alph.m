function [results, optimal_policies] = run_in_sample_QL_alph(combinations, w0, G, T, nw, np, mu_vals, sig_vals, alpha, seed)
    % Function to run in-sample Q-learning for different configurations
    % of epsilon and epochs, and return the results and optimal polcies.

    % Initialize an array to store results and optimal policies
    results = [];
    optimal_policies = cell(size(combinations, 1), 1);

    % Run the function for each specific combination
    for i = 1:size(combinations, 1)
        epsilon = combinations(i, 1);
        epochs = combinations(i, 2);
        alpha = combinations(i, 3);

        % Call the TEC_QL_GBWM_solve_QL function for each combination
        [suc_rl, Q_rl, op_rl] = TEC_QL_GBWM_solve_QL(w0, G, T, nw, np, mu_vals, sig_vals, epochs, epsilon, alpha, seed);
        results = [results; epsilon, epochs, alpha, suc_rl];
        optimal_policies{i} = op_rl;
    end
end