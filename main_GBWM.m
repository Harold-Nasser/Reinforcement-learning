%% Main function to set the problem's parameter and call the function TEC_QL_GBWM_solve_QL
clc
clear
tic
%% Variables
w0 = 100;   % initial wealth
G = 200;    % goal
T = 10;     % horizon
nw = 101;   % nb of wealth points
np = 15;    % nb of portfolios considered
epsi = 0.3; % level of exploration (vs exploitation)
sims = 100000; % number of simulations to run for the out of sample
seed = 1;  % Seed control for the simulations. Requires a positive integer.

% Parameters of the efficient portfolios
mu_vals = [0.0526,0.0552,0.0577,0.0603,0.0629,0.0655,0.0680,0.0706,0.0732,0.0757,0.0783,0.0809,0.0835,0.0860,0.0886];
sig_vals = [0.0485,0.0486,0.0493,0.0508,0.0529,0.0556,0.0587,0.0623,0.0662,0.0705,0.0749,0.0796,0.0844,0.0894,0.0945];

% Combinations of epsilon and epoch for table replication
combinations = [
    0.10, 50000, 0.2;
    0.10, 100000, 0.2;
    0.20, 50000, 0.6; 
    0.20, 100000, 0.6;
    0.25, 100000, 0.8;
    0.30, 50000, 0.8; 
    0.30, 100000, 0.50;
    0.40, 50000,  0.50;
    0.40, 100000, 0.50;
    0.40, 200000, 0.50;
    0.40, 500000, 0.50
];


%% Quick test of the Q-Learning algorithm. Target success rate of professor: 0.6612
epochs = 50000;    % Size of training set
alpha = 0.1;       % Fix learning rate
[suc_rl,Q_rl,op_rl, W] = TEC_QL_GBWM_solve_QL(w0,G,T,nw,np,mu_vals,sig_vals,epochs,epsi,alpha,seed);
fprintf('TEST QBWM ALGORITHM TO MATCH PROFESSOR SUCCESS RATE\n\nTarget success rate: 0.6612\nAchieved success rate: %.4f\n\n******************************************\n\nIN SAMPLE TABLE REPLICATION\n\n', suc_rl);

% Create the heatmap
h = heatmap(op_rl);

% Axis labels and titles
h.XLabel = 'Time Step';
h.YLabel = 'Wealth';
h.Title = 'QL Optimal Policy';

% Setting up the Y axis labels for better legibility
interval = 10;  % Display a label every 10 units

%Put wealth states in the right order for the heatmap
W_inv = flip(W);

% Initialize Y-axis display labels as empty strings
h.YDisplayLabels = repmat({''}, size(W_inv));

% Set every tenth label from the wealth values
for i = 1:interval:length(W_inv)
    h.YDisplayLabels{i} = num2str(W_inv(i));
end

% Include colorbar
h.ColorbarVisible = 'on';


%% In sample table replication
% Run the function which will provide us with the success rates and optimal
% policies for the combinations of epsilon and epoch of the table
[results, optimal_policies] = run_in_sample_QL(combinations, w0, G, T, nw, np, mu_vals, sig_vals, alpha, seed);

% Display the table
% Display table title
disp('MODEL OUTPUTS');

% Display column names
fprintf('%-12s |%-12s |%-12s\n', 'Epsilon', 'No. of epochs', 'V[W(0), t = 0]');

% Display each row of results
for i = 1:size(results, 1)
    fprintf('%-12.2f |%-12d  |%-12.4f\n', results(i, 1), results(i, 2), results(i, 3));
end
fprintf('\n******************************************\n\nOUT OF SAMPLE TESTING\n\n');

%% Test our optimal policy out of sample
[results_oof, success_rates] = test_optimal_policy_multiple(results, optimal_policies, w0, G, T, mu_vals, sig_vals, seed, sims, nw);

% Display the table
% Display table title
disp('MODEL OUTPUTS WITH OUT OF SAMPLE SR');

% Display column names
fprintf('%-12s |%-12s |%-12s |%-12s\n', 'Epsilon', 'No. of epochs', 'V[W(0), t = 0]', 'OOS V[W(0), t = 0]');

% Display each row of results
for i = 1:size(results_oof, 1)
    fprintf('%-12.2f |%-12d  |%-12.4f   |%-12.4f\n', results_oof(i, 1), results_oof(i, 2), results_oof(i, 3), results_oof(i, 4));
end
fprintf('\n******************************************\n\n');

%% Quick test of the SARSA algorithm.
epochs = 50000;    % Size of training set
alpha = 0.1;       % Fix learning rate
[suc_rl_SARSA,Q_rl_SARSA,op_rl_SARSA] = TEC_QL_GBWM_solve_QL_CANEVAS_SARSA(w0,G,T,nw,np,mu_vals,sig_vals,epochs,epsi,alpha,seed);
fprintf('TEST SARSA ALGORITHM\n\nAchieved success rate: %.4f\n\n******************************************\n\nIN SAMPLE TABLE REPLICATION SARSA\n\n', suc_rl_SARSA);

% Create the heatmap
h = heatmap(op_rl_SARSA);

% Axis labels and titles
h.XLabel = 'Time Step';
h.YLabel = 'Wealth';
h.Title = 'SARSA Optimal Policy';

% Setting up the Y axis labels for better legibility
interval = 10;  % Display a label every 10 units

%Put wealth states in the right order for the heatmap
W_inv = flip(W);

% Initialize Y-axis display labels as empty strings
h.YDisplayLabels = repmat({''}, size(W_inv));

% Set every tenth label from the wealth values
for i = 1:interval:length(W_inv)
    h.YDisplayLabels{i} = num2str(W_inv(i));
end

% Include colorbar
h.ColorbarVisible = 'on';

%% In sample table replication SARSA
% Run the function which will provide us with the success rates and optimal
% policies for the combinations of epsilon and epoch of the table
[results_SARSA, optimal_policies_SARSA] = run_in_sample_QL_SARSA(combinations, w0, G, T, nw, np, mu_vals, sig_vals, alpha, seed);

% Display the table
% Display table title
disp('MODEL OUTPUTS');

% Display column names
fprintf('%-12s |%-12s |%-12s\n', 'Epsilon', 'No. of epochs', 'V[W(0), t = 0]');

% Display each row of results
for i = 1:size(results_SARSA, 1)
    fprintf('%-12.2f |%-12d  |%-12.4f\n', results_SARSA(i, 1), results_SARSA(i, 2), results_SARSA(i, 3));
end
fprintf('\n******************************************\n\nOUT OF SAMPLE TESTING SARSA\n\n');


%% Test our optimal policy out of sample for SARSA
[results_oof_SARSA, success_rates_SARSA] = test_optimal_policy_multiple(results_SARSA, optimal_policies_SARSA, w0, G, T, mu_vals, sig_vals, seed, sims, nw);

% Display the table
% Display table title
disp('MODEL OUTPUTS WITH OUT OF SAMPLE SR');

% Display column names
fprintf('%-12s |%-12s |%-12s |%-12s\n', 'Epsilon', 'No. of epochs', 'V[W(0), t = 0]', 'OOS V[W(0), t = 0]');

% Display each row of results
for i = 1:size(results_oof_SARSA, 1)
    fprintf('%-12.2f |%-12d  |%-12.4f   |%-12.4f\n', results_oof_SARSA(i, 1), results_oof_SARSA(i, 2), results_oof_SARSA(i, 3), results_oof_SARSA(i, 4));
end
fprintf('\n******************************************\n\n');
%%  Test avec différent niveau exploitation et parametre 
sims = 100000;
combinations = [
    0.10, 50000, 0.15;
    0.10, 100000, 0.15;
    0.20, 50000, 0.2; 
    0.20, 100000, 0.2;
    0.25, 100000, 0.005;
    0.30, 50000, 0.07; 
    0.30, 100000, 0.07;
    0.40, 50000,  0.05;
    0.40, 100000, 0.05;
    0.40, 200000, 0.05;
    0.40, 500000, 0.05
];

[results_alph, optimal_policies_alph] = run_in_sample_QL_alph(combinations, w0, G, T, nw, np, mu_vals, sig_vals, alpha, seed);
[results_oof_alph, success_rates_alph] = test_optimal_policy_multiple(results_alph, optimal_policies_alph, w0, G, T, mu_vals, sig_vals, seed, sims, nw);

% Display the table
% Display table title
disp('MODEL OUTPUTS Q-Learning variation parametres');

% Display column names
 fprintf('%-12s |%-12s |%-12s |%-12s\n', 'Epsilon', 'No. of epochs',  'Alpha', 'V[W(0), t = 0]' );
% Display each row of results
for i = 1:size(results_oof_alph, 1)
    fprintf('%-12.2f |%-12d |%-12.4f |%-12.2f\n', results_oof_alph(i, 1), results_oof_alph(i, 2), results_oof_alph(i, 3), results_oof_alph(i, 4)); 
    
end
fprintf('\n******************************************\n\nOUT OF SAMPLE TESTING \n\n');
 
%% Test our optimal policy out of sample for SARSA

combinations = [
    0.10, 50000, 0.15;
    0.10, 100000, 0.15;
    0.20, 50000, 0.2; 
    0.20, 100000, 0.2;
    0.25, 100000, 0.005;
    0.30, 50000, 0.07; 
    0.30, 100000, 0.07;
    0.40, 50000,  0.05;
    0.40, 100000, 0.05;
    0.40, 200000, 0.05;
    0.40, 500000, 0.05
];


[results_SARSA_alph, optimal_policies_SARSA_alph] = run_in_sample_QL_SARSA_alph(combinations, w0, G, T, nw, np, mu_vals, sig_vals, alpha, seed);

[results_oof_SARSA_alph, success_rates_SARSA_alph] = test_optimal_policy_multiple(results_SARSA_alph, optimal_policies_SARSA_alph, w0, G, T, mu_vals, sig_vals, seed, sims, nw);

% Display the table
% Display table title
disp('MODEL OUTPUTS WITH OUT OF SAMPLE SR variation parametre');

% Display column names
 fprintf('%-12s |%-12s |%-12s |%-12s\n', 'Epsilon', 'No. of epochs',  'Alpha', 'V[W(0), t = 0]' );

% Display each row of results
for i = 1:size(results_oof_SARSA_alph, 1)
    fprintf('%-12.2f |%-12d |%-12.4f |%-12.2f\n', results_oof_SARSA_alph(i, 1), results_oof_SARSA_alph(i, 2), results_oof_SARSA_alph(i, 3), results_oof_SARSA_alph(i, 4)); 
end
% fprintf('\n******************************************\n\n');
 toc 




