function errors = F_IncrementLinearEstimation4(X, Y, Xtest, Ytest, dS, ...
    lambda, Type)

p = length(dS);
[Ny, M] = size(Ytest);
errors = zeros(p, 1);

if strcmp(Type, 'RMSE')
    eref = max([sqrt(M-1); 1]);
elseif strcmp(Type, 'NMSE')
    eref = norm(Ytest(:));
end

lambda = lambda * size(Y, 2);
X     = X(dS, :);
Xtest = Xtest(dS, :);

PXX = X * X' + lambda * eye(p);
PYX = Y * X';

Omega = zeros(Ny, p);
Theta = zeros(p, M);
GammaS = 1 / PXX(1, 1);
Omega(:, 1) = PYX(:, 1) * sqrt(GammaS);
Theta(1, :) = sqrt(GammaS) * Xtest(1, :);
for k = 1:p-1
    PXX_Sk = PXX(1:k, k+1);
    v = [-GammaS * PXX_Sk; 1];
    d = PXX(k+1, k+1) - PXX_Sk' * GammaS * PXX_Sk;
    GammaS = [GammaS, zeros(k, 1); zeros(1, k), 0] + v * v' / d;
    Omega(:, k+1) = PYX(:, 1:k+1) * v / sqrt(d);
    Theta(k+1, :) = v' * Xtest(1:k+1, :) / sqrt(d);
end

a = veciprod(Theta, Omega' * Ytest);
b = (vecnorm(Omega, 2, 1)' .* vecnorm(Theta, 2, 2)) .^ 2;
Omega2 = Omega' * Omega;
Theta2 = Theta * Theta';
e0 = norm(Ytest(:))^2;

e = e0;
for k = 1:p
    e = e + b(k) - 2 * a(k) + 2 * Omega2(k, 1:k-1) * Theta2(1:k-1, k);
    errors(k) = e;
end

errors = sqrt(errors) / eref;

end

function c = veciprod(A, B)

c = sum(A .* B, 2);

end