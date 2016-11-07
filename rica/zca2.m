function [Z] = zca2(x)
epsilon = 1e-4;
% You should be able to use the code from your PCA/ZCA exercise
% Retain all of the components from the ZCA transform (i.e. do not do
% dimensionality reduction)

% x is the input patch data of size
% z is the ZCA transformed data. The dimenison of z = x.

%%% YOUR CODE HERE %%%

%%================================================================
x = bsxfun(@minus,x,mean(x,1));
sigma = x * x' / (size(x,2));
[U, S, V] = svd(sigma);

diagS = diag(S);
r = nnz(diagS);
for k = 1:r
   var_retain = sum(diagS(1:k))/sum(diagS(1:r));
   if var_retain>=0.99
       break;
   end 
end
xPCAWhite = diag(1./sqrt(diagS(1:k) + epsilon)) * U(:,1:k)' * x;
Z = U(:, 1:k) * xPCAWhite;

