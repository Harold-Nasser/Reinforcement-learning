function [success, Q, optimal_policy] = TEC_QL_GBWM_solve_QL_CANEVAS_SARSA(w0,G,T,n_states,n_pf,mu_vals,sig_vals,epochs,epsilon,alpha,seed)
% Q-learning approach to solve a Goal-based Wex(2)alth Management problem

rng(seed);  % Seed control for the simulations

% Grid of wealth, including cumulative transition probabilities
[W,start_idx,start_ctpm,ctpm] = TEC_QL_GBWM_build_grid(w0,G,T,mu_vals,sig_vals,n_states,3);

%Initialize Q-table
Q = zeros(n_states,T+1,n_pf);

for epoch = 1:epochs
   idx = start_idx;
  % choose action with epsilon-greedy policy
       a0 = get_action(Q,idx,1,epsilon,n_pf);
   for t=1:(T+1)
         
       tnext = t+1;
       if t < (T+1) 
          % Random choice of the next state, using the appropriate transitions.
          if t==1
              idx1 = sum( start_ctpm(:,a0) < rand(1,1) ) + 1;
          else  % t > 1  
              idx1 = sum( ctpm(idx,:,a0) < rand(1,1) ) + 1;
          end
           % Choose action for the next state
        a1 = get_action(Q,idx1, tnext, epsilon, n_pf);
          % SARSA Q-factor update : 
          Q(idx,t,a0)=alpha*  Q(idx1,tnext,a1) + (1-alpha )*Q(idx,t,a0);
           
         
      else   % i.e. t==T+1, the end of the horizon
          % Is the goal reached ?
        reward = (W(idx) >= G);
           
          % Q-factor update :
          Q(idx, T+1, a0) = Q(idx, T+1, a0) + alpha * (reward - Q(idx, T+1, a0));
  
            %idx1 = idx;  % Command seems redundant, MD 11 2023
           
       end
       idx = idx1(1);   % The (1) seems redundant, MD 11 2023
       a0=a1;
   end
end
% Compute success rate and optimal policy
success = max(Q(start_idx,1,:));
[~, optimal_policy] = max(Q, [], 3);

end

% get_action uses an epsilon-greedy policy to select the next action
function [action] = get_action(Q,idx,t,epsilon,n_pf)

if rand(1,1) < epsilon
   % Call randomaction to pick a random action
   action = randomaction(n_pf);
else 
   q = Q(idx,t,:);
   max_idx = squeeze(double(q==max(q)))';
   
   if sum(max_idx,2)>1
      possible_actions = nonzeros(max_idx.*double(1:n_pf));
      action_idx = randomaction(length(possible_actions));
      action = possible_actions(action_idx);
   else
       [~,action] = max(q);
   end
end
end

% randomaction function
function action = randomaction(n_pf)
    vals = cumsum(ones(1,n_pf)*(1/n_pf));
    rnd = rand(1,1);
    action = sum(vals < rnd) + 1;
end
