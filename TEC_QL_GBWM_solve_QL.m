function [success, Q, optimal_policy, W] = TEC_QL_GBWM_solve_QL(w0, G, T, n_states, n_pf, mu_vals, sig_vals, epochs, epsilon, alpha, seed)
    % Implements Q-learning for goal-based wealth management.
    % w0: Initial wealth
    % G: Wealth goal at time T
    % T: Time horizon
    % n_states: Number of wealth levels to consider
    % n_pf: Number of portfolio options
    % mu_vals, sig_vals: Arrays of expected returns (mu) and volatilities (sigma) for each portfolio
    % epochs: Number of iterations for the learning process
    % epsilon: Parameter for epsilon-greedy policy (balance between exploration and exploitation)
    % alpha: Learning rate
    % seed: Random seed for reproducibility

    rng(seed);  % Sets the seed for random number generation for consistent results

    % Generates grid for wealth levels and transition probabilities
    [W, start_idx, start_ctpm, ctpm] = TEC_QL_GBWM_build_grid(w0, G, T, mu_vals, sig_vals, n_states, 3);
    % Initialize the Q-table with dimensions: states x time steps x portfolios
    Q = zeros(n_states, T+1, n_pf);

    % Iterate over each epoch (learning iteration)
    for epoch = 1:epochs
        idx = start_idx;  % Start from the initial wealth state
        for t = 1:(T+1)  % Iterate over each time step
            % Choose an action based on the current Q-table using an epsilon-greedy approach
            a0 = get_action(Q, idx, t, epsilon, n_pf);

            tnext = t + 1;  % Next time step
            if t < T+1  % For all time steps except the last one
                % Determine the next state based on the current state, action, and transition probabilities
                % Get the next amount of wealth based off current state and
                % action and trasnsition probabilities
                if t == 1
                    idx1 = sum(start_ctpm(:, a0) < rand(1, 1)) + 1;
                else  
                    idx1 = sum(ctpm(idx, :, a0) < rand(1, 1)) + 1;
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Calculate the maximum future Q-value for the next state
                V_next = max(Q(idx1, tnext, :));
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Update the Q-value for the current state-action pair
                Q(idx, t, a0) = alpha*V_next + (1-alpha) * (Q(idx, t, a0));
                else  % At the last time step
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Check if the wealth goal is reached
                is_goal_reached = double((W(idx) >= G));
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Update the Q-value for reaching or not reaching the goal
                Q(idx, t, a0) = Q(idx, t, a0) + alpha * (is_goal_reached - Q(idx, t, a0));
                idx1 = idx;  % Command seems redundant, MD 11 2023
            end
            idx = idx1(1);  % Update the current state for the next iteration
        end
    end

    % Compute the success rate of reaching the goal and derive the optimal policy from the Q-table
    success = max(Q(start_idx, 1, :));
    [~, optimal_policy] = max(Q, [], 3);  % Optimal policy is the action with the highest Q-value at each state
end

function [action] = get_action(Q, idx, t, epsilon, n_pf)
    % Selects an action using an epsilon-greedy policy based on the current Q-table
    % Q: Current Q-table
    % idx: Current state index
    % t: Current time step
    % epsilon: Probability of choosing a random action (exploration)
    % n_pf: Number of portfolio options (actions)

    if rand(1, 1) < epsilon
        % With probability epsilon, choose a random action for exploration
        action = randomaction(n_pf);
    else 
        % Otherwise, choose the best action based on the current Q-values (exploitation)
        q = Q(idx, t, :);
        max_idx = squeeze(double(q == max(q)))';

        % In case of multiple actions with the same Q-value, randomly choose one of them
        if sum(max_idx, 2) > 1
            possible_actions = nonzeros(max_idx .* double(1:n_pf));
            action_idx = randomaction(length(possible_actions));
            action = possible_actions(action_idx);
        else
            % If there is a single best action, choose it
            [~, action] = max(q);
        end
    end
end

function action = randomaction(n_pf)
    % Returns a random action (portfolio choice)
    % n_pf: Number of portfolio options
    action = randi(n_pf);  % Randomly select an integer between 1 and n_pf
end
