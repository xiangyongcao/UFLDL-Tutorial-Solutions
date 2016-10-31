function [f,g] = softmax_regression_vec(theta, X,y)
%
% Arguments:
%   theta - A vector containing the parameter values to optimize.
%       In minFunc, theta is reshaped to a long vector.  So we need to
%       resize it to an n-by-(num_classes) matrix.
%
%   X - The examples stored in a matrix.
%       X(i,j) is the i'th coordinate of the j'th example.
%   y - The label for each example.  y(j) is the j'th example's label.
% n: the dimension of theta plus 1
% m : the number of samples
[n, m] = size(X);
K = length(unique(y));

% theta is a vector;  need to reshape to a matrix: n x K
theta = reshape(theta, n, K);
groundTruth = full(sparse(y, 1:m, 1))'; % m x K
temp = theta' * X;  % K x m
M = bsxfun(@minus,temp,max(temp, [], 1));  % K x m
M = exp(M);  
p = bsxfun(@rdivide, M, sum(M));   % K x m

% compute objective function value
f = -sum(sum(groundTruth'.*log(p)));

% compute gradient
g = zeros(size(theta));
g = -X * (groundTruth - p');
g = g(:); % make gradient a vector for minFunc

