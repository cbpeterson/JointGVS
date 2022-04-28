function [gamma_save, Omega_save, adj_save, ar_gamma, info] = MCMC_linear_model_with_graph_selection(X, Y, Z, ...
    a_0, b_0, h_alpha, h_beta, a, b, v0, v1, lambda, pii, ...
    gamma, burnin, nmc, summary_only)

% Standardize data so that S(i,i) = n 
S = X' * X;
[n, p] = size(X);
S = corrcoef(S) * n;

% Initial guess for Sigma, precision matrix, and adjacency matrix
Sig = eye(p);
Omega = inv(Sig);
adj = eye(p);

V0 = v0 * ones(p);
V1 = v1 * ones(p);

tau = V1;
ind_noi_all = zeros(p-1, p);

for i = 1:p
    if i==1
        ind_noi = [2:p]';
    elseif i==p
        ind_noi = [1:p-1]';
    else
        ind_noi = [1:i-1,i+1:p]';
    end
    ind_noi_all(:,i) = ind_noi;
    
end

pii_RB = zeros(p);
pii_mat = zeros(p);

% Always keep variable selections
gamma_save = zeros(p, nmc);

% Record some diagnostic info
full_gamma_save = zeros(p, burnin + nmc);
node_degrees = zeros(p, burnin + nmc);

% Keep track of info to compute acceptance rates
n_gamma_prop = 0;
n_gamma_accept = 0;
n_add_prop = 0;
n_add_accept = 0;
n_remove_prop = 0;
n_remove_accept = 0;

% Allocate storage for MCMC sample, or just for means if only summary is
% required
if summary_only
    Omega_save = zeros(p, p);
    adj_save = Omega_save;
else
    Omega_save = zeros(p, p, nmc);
    adj_save = Omega_save;
end

% Number of currently included variables
p_gamma = sum(gamma);

% MCMC sampling
for iter = 1: burnin + nmc
 
    % Print out info every 100 iterations
    if mod(iter, 100) == 0
        fprintf('Iteration = %d\n', iter);
        fprintf('Number of included variables = %d\n', sum(gamma));
        fprintf('Number of add variable moves proposed %d and accepted %d\n', n_add_prop, n_add_accept);
        fprintf('Number of remove variable moves proposed %d and accepted %d\n', n_remove_prop, n_remove_accept);
        fprintf('Number of included edges %d \n\n', (sum(sum(adj)) - p) / 2);
    end
    
    % Select an entry at random to toggle
    change_index = randsample(p, 1);
    
    % Keep track of number of add vs. remove moves proposed
    if (gamma(change_index) == 0) 
        n_add_prop = n_add_prop + 1;
    else
        n_remove_prop = n_remove_prop + 1;
    end
    n_gamma_prop = n_gamma_prop + 1;
    
    % Toggle value
    gamma_prop = gamma;
    gamma_prop(change_index) = abs(gamma(change_index) - 1);

    % Compute MH ratio on log scale
    log_r = log_r_y(gamma, gamma_prop, X, Y, Z, 0, h_alpha, h_beta, a_0, b_0) + ...
        log_r_gamma_given_G(gamma, gamma_prop, adj, a, b);
    
    % Accept proposal with probability r
    if (log(rand(1)) < log_r)
        if (gamma(change_index) == 0) 
            gamma(change_index) = 1;
            p_gamma = p_gamma + 1;
            n_add_accept = n_add_accept + 1;
        else
            gamma(change_index) = 0;
            p_gamma = p_gamma - 1;
            n_remove_accept = n_remove_accept + 1;
        end
        n_gamma_accept = n_gamma_accept + 1;
    end
    
    % Resample precision matrix and graph
    %%% sample Sig and Omega = inv(Sig)
    for i = 1:p
        ind_noi = ind_noi_all(:,i);
        tau_temp = tau(ind_noi,i);
        
        Sig11 = Sig(ind_noi,ind_noi);
        Sig12 = Sig(ind_noi,i);
        
        invC11 = Sig11 - Sig12*Sig12'/Sig(i,i);
        
        Ci = (S(i,i)+lambda)*invC11+diag(1./tau_temp);
        Ci = (Ci+Ci')./2;
        Ci_chol = chol(Ci);
        mu_i = -Ci_chol\(Ci_chol'\S(ind_noi,i));
        beta = mu_i+ Ci_chol\randn(p-1,1);

        Omega(ind_noi,i) = beta;
        Omega(i,ind_noi) = beta;
        
        a_gam = 0.5*n+1;
        b_gam = (S(i,i)+lambda)*0.5;
        gam = gamrnd(a_gam,1/b_gam);
        
        c = beta'*invC11*beta;
        Omega(i,i) = gam+c;
        
        %% Below updating Covariance matrix according to one-column change of precision matrix
        invC11beta = invC11*beta;
        
        Sig(ind_noi,ind_noi) = invC11+invC11beta*invC11beta'/gam;
        Sig12 = -invC11beta/gam;
        Sig(ind_noi,i) = Sig12;
        Sig(i,ind_noi) = Sig12';
        Sig(i,i) = 1/gam;  
        
        v0 = V0(ind_noi,i);
        v1 = V1(ind_noi,i);
        
        w1 = -0.5*log(v0) -0.5*beta.^2./v0+log(1-pii);
        w2 = -0.5*log(v1) -0.5*beta.^2./v1+log(pii);
        
        w_max = max([w1,w2],[],2);
        
        w = exp(w2-w_max)./sum(exp([w1,w2]-repmat(w_max,1,2)),2);
        
        z = (rand(p-1,1)<w);
        
        v = v0;
        v(z) = v1(z);
        
        pii_mat(ind_noi,i) = w;
        
        tau(ind_noi,i) = v;
        tau(i,ind_noi) = v;
        
        adj(ind_noi,i) = z;
        adj(i,ind_noi) = z; 
    end
    
    if iter > burnin
        gamma_save(:, iter-burnin) = gamma;
        pii_RB = pii_RB + pii_mat/nmc;

        if summary_only
            Omega_save(:, :) = Omega_save(:, :) + Omega / nmc;
            adj_save(:, :) = adj_save(:, :) + adj / nmc;
        else
            Omega_save(:, :, iter-burnin) = Omega;
            adj_save(:, :, iter-burnin) = adj;
        end
    end
    
    full_gamma_save(:, iter) = gamma;
    node_degrees(:, iter) = sum(adj, 2) - 1;

end

ar_gamma = n_gamma_accept / n_gamma_prop;

% Info for diagnostic purposes
info = struct('n_add_prop', n_add_prop, 'n_add_accept', n_add_accept, ...
    'n_remove_prop', n_remove_prop, 'n_remove_accept', n_remove_accept, ...
    'full_gamma', full_gamma_save, ...
    'node_degrees', node_degrees);