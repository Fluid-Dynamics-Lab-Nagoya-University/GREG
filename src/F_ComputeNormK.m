function normK = F_ComputeNormK(X, Y, S, lambda)

lambda = lambda * size(X, 2);
XS = X(S, :);
normK = norm(Y * XS' / (XS * XS' + lambda * eye(length(S))), 'fro');

end