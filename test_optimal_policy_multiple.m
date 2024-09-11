function [results_with_success_rate, success_rates] = test_optimal_policy_multiple(results, optimal_policies, w0, G, T, mu_vals, sig_vals, seed, num_simulations, nw)
    % Function to test each optimal policy out of sample and append the success rate to the results.

    rng(seed);  % Set the random seed
    success_rates = zeros(size(results, 1), 1);
    
     % Generate the grid of wealth levels using your function
    [W, ~] = TEC_QL_GBWM_build_grid(w0, G, T, mu_vals, sig_vals, nw, 3);
    
    % Run simulations for each optimal policy
    for i = 1:size(optimal_policies, 1)
        num_successes = 0;
        for sim = 1:num_simulations
            current_wealth = w0;
            for t = 1:T
                [~, state_idx] = min(abs(W - current_wealth)); % Find current wealth state
                chosen_action = optimal_policies{i}(state_idx, t); % Choose action based on policy

                % Simulate return and update wealth
                mu = mu_vals(chosen_action);
                sigma = sig_vals(chosen_action);
                simulated_return = mu + sigma * randn(); % Assuming normal distribution of returns
                current_wealth = current_wealth * exp(simulated_return);
            end

            % Check if goal is reached
            if current_wealth >= G
                num_successes = num_successes + 1;
            end
        end

        % Calculate success rate for this policy
        success_rate = (num_successes / num_simulations);
        success_rates(i) = success_rate;
    end

    % Append success rates to the results
    results_with_success_rate = [results, success_rates];
end



