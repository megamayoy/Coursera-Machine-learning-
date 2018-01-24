function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%



%possible value for both C and sigma

pos_values = [0.01 0.03 0.1 0.3  1 3 10 30];

min_error = 1;

for i = 1:8
 
 for j = 1:8
 
    % train with training set and evaluate using the cross validation
    %get the least error C and sigma
    current_sigma = pos_values(i);
    current_c = pos_values(j);
    %train the model and get THETAs x1,x2 are dummy varaibles.
    model = svmTrain(X, y, current_c, @(a, b) gaussianKernel(a, b,current_sigma));
 
    % evaluate   
    predictions = svmPredict(model, Xval);
    
    current_error = mean(double(predictions ~= yval) );
    
    if (current_error < min_error)
      
         min_error = current_error;
         C = current_c;
         sigma = current_sigma;
      
    endif
    
 end
end


% =========================================================================

end
