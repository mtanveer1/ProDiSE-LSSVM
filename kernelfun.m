function [omega] = kernelfun(Xtrain,kerfPara,Xt)

% function [K] = kernelfun(M,kerfPara,X)

% Construct the positive (semi-) definite and symmetric kernel matrix
%
% >> Omega = kernelfun(X, kernel_fct, sig2)
%
% This matrix should be positive definite if the kernel function
% satisfies the Mercer condition. Construct the kernel values for
% all test data points in the rows of Xt, relative to the points of X.
%
% >> Omega_Xt = kernelfun(X, kernel_fct, sig2, Xt)
%
%
% Full syntax
%
% >> Omega = kernelfun(X, kernel_fct, sig2)
% >> Omega = kernelfun(X, kernel_fct, sig2, Xt)
%
% Outputs
%   Omega  : N x N (N x Nt) kernel matrix
% Inputs
%   X      : N x d matrix with the inputs of the training data
%   kernel : Kernel type (by default 'RBF_kernel')
%   sig2   : Kernel parameter (bandwidth in the case of the 'RBF_kernel')
%   Xt(*)  : Nt x d matrix with the inputs of the test data
kernel_type = kerfPara.type;
kernel_pars = kerfPara.pars;

nb_data = size(Xtrain,1);


%%%%%%%%%%%%%calculating kernel matrix
if strcmp(kernel_type,'rbf')
    if nargin<3
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        omega = exp(-omega./(2*kernel_pars(1)^2));
    else
        omega = - 2*Xtrain*Xt';
        Xtrain = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        Xt = sum(Xt.^2,2)*ones(1,nb_data);
        omega = omega + Xtrain+Xt';
        omega = exp(-omega./(2*kernel_pars(1)^2));
    end
    
    
end