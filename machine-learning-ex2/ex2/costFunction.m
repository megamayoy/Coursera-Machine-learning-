function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
num_of_features = length(theta); %num_of_thetas
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

hypothesis_function = sigmoid(X*theta);

J = sum(-y.*log(hypothesis_function) - (1-y).*log(1-hypothesis_function))/m;




  calculate_error = hypothesis_function - y;
 
  new_matrix = calculate_error.*X;
  
  
  
  for i = 1:num_of_features
   
   grad(i,1) = sum(new_matrix(:,i))/m;
   
  end 


% =============================================================

end
