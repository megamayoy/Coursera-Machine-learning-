function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
%this is where the mean of each feature will be stored 
mu = zeros(1, size(X, 2));
%this is where the standard deviation of each feature will be stored
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       
%declare the number of features in the dataset
num_of_features = size(X,2);


%computing the mean of each feature (this is a row vector)
mu = mean(X);
%computing the standard deviation of each feature (this is a row vector)

sigma = std(X);

%remember that we subtract the mean first then we divide by the standard deviation
for i = 1:num_of_features

 X_norm(:,i) = (X_norm(:,i)- mu(1,i))./sigma(1,i);    


endfor


% ============================================================

end
