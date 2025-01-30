function [e, Yest, A] = F_LinearEstimation(X, Y, Xtest, Ytest, S, lambda, Type)

if strcmp(Type, 'RMSE')
    eref = sqrt(size(Ytest, 2)-1);
elseif strcmp(Type, 'NMSE')
    eref = norm(Ytest(:));
end

lambda = lambda * size(Y, 2);

XS = X(S, :);
A = Y * XS' / (XS * XS' + lambda * eye(length(S)));
Yest = A * Xtest(S, :);
e = norm(Ytest - Yest, 'fro') / eref;

end