% Set random number seed forreproducibility
rng(1717);

% Read in data from file - X should be the predictor matrix (n x p), and
% is the response variable (n x 1)
% Note that X should be column centered (i.e. the means of the columns in X
% should be 0), and you will need to modify how you read in the data
% if you have row or column names in the csv file
X = csvread('my_xmatrix.csv');
Y = csvread('my_y.csv');

% Number of variables and number of observations
p = size(X, 2);
n = size(X, 1);

% Here I am using using 0s for the Z (i.e. no covariates), but you could
% sub in real covariates here (such as age and gender) that are not subject
% to selection. If using no covariates, set h_alpha to 0 below
Z = zeros(n, 5);

% Set prior parameters

% Shape and scale of inverse gamma prior on tau^2
a_0 = 3;
b_0 = 0.5;

% Parameters of MRF prior - how to determine proper settings for a and b?
% Li and Zhang discuss this, especially the phase transition property
% a should be negative, with values farther from 0 corresponding to sparser
% variable selection
b = 0.5;
a = -2.75;

% Note that the parameterization used in the code is slightly different from those in Wang (2014).
% (h in code) =  (h in paper )^2
h = 100^2;

% (v0 in code) = (v0 in paper)^2
v0 = 0.1^2;

% (v1 in code) = (v1 in paper)^2
v1 = h * v0;

% Remaining parameters for graph selection
lambda = 1;
pii = 2 / (p - 1);

% These parameter settings are for standardized covariates
h_alpha = 1;
h_beta = 1;

% Initial value of gamma (variable selection indicators)
gamma_init = zeros(p, 1);

% Number of burnin and post-burnin iterations
burnin = 10000;
nmc = 10000;

% Run MCMC sampler for joint graph and variable selection
% Clinical covariates Z are set to all zeros here (as in simulation from
% paper)
% Since p is large, param summary_only is set to true. This means that
% Omega_save and adj_save will be the MCMC averages rather than the full
% set of sample values from each iteration
tic
[gamma_save, Omega_save, adj_save, ar_gamma, info] = ...
    MCMC_linear_model_with_graph_selection(X, Y, Z, ...
    a_0, b_0, h_alpha, h_beta, a, b, v0, v1, lambda, pii, ...
    gamma_init, burnin, nmc, true);
toc

% Write out some diagnostic summaries
% Number of proposals to add variables, number accepted, number of
% proposals to remove, and number accepted
csvwrite('mh_info.csv', ...
    [info.n_add_prop', info.n_add_accept' ...
    info.n_remove_prop', info.n_remove_accept']);
% Variable selection indicators for all iterations
csvwrite('full_gamma.csv', info.full_gamma');
% Node degrees in all iterations
csvwrite('node_degrees_model.csv', info.node_degrees');

% Summarize graph structure learning
ppi_var = mean(gamma_save, 2);
ppi_edges = adj_save;

% Edges selected using marginal PPI threshold of 0.5 as a symmetric matrix
sel_edges = (ppi_edges > 0.5) - eye(p);


