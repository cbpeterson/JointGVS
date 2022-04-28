function [log_y_mh_ratio] = log_r_y(gamma, gamma_prop, X, Y, Z, h_0, h_alpha, h_beta, a_0, b_0)
% Compute MH ratio p(Y|gamma_prop) / p(Y|gamma) on log scale

[n, p] = size(X);
X_gamma = X(:, find(gamma));
X_gamma_prop = X(:, find(gamma_prop));

% Use logdet rather than log(det()) is case det is large/small
% Similarly, log1p computes log(1 + p) which is accurate for small p
core_term = eye(n) + h_0 * (ones(n, 1) * ones(1, n)) + h_alpha * (Z * Z') + h_beta * (X_gamma * X_gamma');
core_term_prop = eye(n) + h_0 * (ones(n, 1) * ones(1, n)) + h_alpha * (Z * Z') + h_beta * (X_gamma_prop * X_gamma_prop');
log_y_mh_ratio = 0.5 * logdet(core_term, 'chol') + ...
    (n + 2 * a_0) / 2 * log1p(1 / 2 / b_0 * Y' * inv(core_term) * Y) - ...
    0.5 * logdet(core_term_prop, 'chol') - ...
    (n + 2 * a_0) / 2 * log1p(1 / 2 / b_0 * Y' * inv(core_term_prop) * Y);
end