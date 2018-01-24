function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
num_of_features = length(theta); %num_of_thetas
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

  hypothesis_function = X*theta;
 
  calculate_error = hypothesis_function - y;
 
  new_matrix = calculate_error.*X;
  % we can call it a slope or a gradient
  slopes = ones(num_of_features,1);
  
  for i = 1:num_of_features
   
   slopes(i,1) = sum(new_matrix(:,i))/m;
   
  end 
   theta = theta - (alpha* slopes);

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end



end
