function [f,g] = logistic_regression_vec(theta, X,y)
%
% Arguments:
%   theta - A column vector containing the parameter values to optimize.
%   X - The examples stored in a matrix.
%       X(i,j) is the i'th coordinate of the j'th example.
%   y - The label for each example.  y(j) is the j'th example's label.

% compute objective function value and gradient
temp = theta'*X;
y_hat = sigmoid(temp);
f = - sum(y .* log(y_hat) + (1 - y) .* log(1-y_hat));
g = X * (y_hat - y)';

% epsilon = 5*1e-2;
% j = randsample(length(theta),1);
% theta0 = theta; theta0(j) = theta0(j) - epsilon;
% theta1 = theta; theta1(j) = theta1(j) + epsilon;
% 
% temp = theta0'*X;
% y_hat = sigmoid(temp);
% f0 = - sum(y .* log(y_hat) + (1 - y) .* log(1-y_hat));
% 
% temp = theta1'*X;
% y_hat = sigmoid(temp);
% f1 = - sum(y .* log(y_hat) + (1 - y) .* log(1-y_hat));
% 
% g_est = (f1 - f0)/(2*epsilon);
% error = abs(g(j) - g_est);
% disp(['gradient is ',num2str(g(j)),' and ',num2str(g_est)]);
% disp(['Absolute error is ',num2str(error)]);
