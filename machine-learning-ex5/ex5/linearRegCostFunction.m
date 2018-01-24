function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


hypothesis = X * theta;

error_dif = (hypothesis - y);

error_calc = sum(error_dif.^2);

regularized_part = sum(theta(2:end,:).^2) * lambda;
J = (error_calc + regularized_part) / (2 * m);

%claculating the gradient 

error_dif_mul_X = error_dif .* X;
if(size(error_dif_mul_X,1) != 1)
error_dif_mul_X = sum(error_dif_mul_X);
endif

error_dif_mul_X = error_dif_mul_X' ./ m;

regularized_part = theta .* (lambda / m);

grad = error_dif_mul_X;

grad(2:end,:) = grad(2:end,:) + regularized_part(2:end,:) ;

% =========================================================================

grad = grad(:);

end
