function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
%sum od thetas squared excluding theta1
sum_of_thetas = sum(theta(2:end,1).^2);

%notice that when i wrote 2*m without the parantheses I got zero when I submitted.
J = sum(-y.*log(hypothesis_function) - (1-y).*log(1-hypothesis_function))/m+((lambda/(2*m))*sum_of_thetas);




  calculate_error = hypothesis_function - y;
 
  new_matrix = calculate_error.*X;
  
  
  
  for i = 1:num_of_features
  
   if i == 1 
   grad(i,1) = sum(new_matrix(:,i))/m;
   
   else   
   grad(i,1) = sum(new_matrix(:,i))/m + ((lambda/m) * theta(i,1));
   
   end
   
  end 


% =============================================================

end
