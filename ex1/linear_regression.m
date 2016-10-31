function [f,g] = linear_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize: n x 1
  %   X - The examples stored in a matrix: n x m, each column is a sample
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The target value for each example: m x 1  
  %       y(j) is the target for example j.
  %
  
 [n m] = size(X);

  f=0;
  g=zeros(size(theta));

  %
  % TODO:  Compute the linear regression objective by looping over the examples in X.
  %        Store the objective function value in 'f'.
  %
  % TODO:  Compute the gradient of the objective with respect to theta by looping over
  %        the examples in X and adding up the gradient for each example.  Store the
  %        computed gradient in 'g'.
  
%%% YOUR CODE HERE %%%
% compute the objective function value
for i = 1:m
    res = 0.5* (theta' * X(:,i) - y(i))^2;
    f = f + res;
end

% compute the gradient of the objective function
for i = 1:m
   temp = X(:,i) .* (theta' * X(:,i) - y(i));
   g = g + temp;
end
