function [log_MH] = log_r_gamma_given_G(gamma, gamma_prop, adj, a, b)
% Compute MH ratio for adding or removing one var

p = size(gamma, 1);

% +1 if adding, -1 if removing a var
gamma_diff = sum(gamma_prop - gamma);

% Assumption in paper is that adjacency matrix has 0's along the diagonal,
% while here is has 1's, so need to subtract eye(p)
adj = adj - eye(p);
    
log_MH = gamma_diff * a + b * (gamma_prop' * adj * gamma_prop - ...
  gamma' * adj * gamma);

end