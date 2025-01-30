function [S, time] = F_GREG(X, Y, p, lambda)

tic
    
Nx = size(X, 1);
Ny = size(Y, 1);
M  = size(X, 2);
S = zeros(p, 1);
time = zeros(p, 1);

lambda = lambda * size(Y, 2);

if Ny > M
    Y = qr(Y, "econ"); % O(Ny M min(Ny, M)), Ref. Matrix Computation, p.278
%     [~, Sigma, V] = svd(Y, "econ");
%     Y = diag(Sigma) .* V';
    PXY = X * Y'; % O(Nx Ny M)
    f = vecnorm(PXY, 2, 2) .^ 2; % O(Nx Ny M)
else
    PXY = X * Y'; % O(Nx Ny M)
    f = vecnorm(PXY, 2, 2) .^ 2; % O(Nx Ny)
end
g = vecnorm(X, 2, 2) .^ 2 + lambda; % O(Nx M)

Omega = zeros(Nx, p-1);
Theta = zeros(size(Y, 1), p-1);
logicalS = false(Nx, 1);
for pp = 1:p
    v = f ./ g;
    v(g < 1e-8 | logicalS) = 0;
    [~, Spp] = max(v);
    S(pp) = Spp;
    time(pp) = toc;
    logicalS(Spp) = true;

    Omega_tmp = Omega(:, 1:pp-1);
    Theta_tmp = Theta(:, 1:pp-1);

    delta = X * X(Spp, :)' - Omega_tmp * Omega_tmp(Spp, :)'; % O(Nx M + Nx k)
    delta(Spp) = delta(Spp) + lambda;
    omega = delta / sqrt(delta(Spp));
    theta = (Y * X(Spp, :)' - Theta_tmp * Omega_tmp(Spp, :)') ...
        / sqrt(delta(Spp)); % O(Ny M + Ny k)

    f = f + omega .* (2 * Omega_tmp * (Theta_tmp' * theta) ...
            - 2 * PXY * theta + norm(theta)^2 * omega); % O((Nx + Ny) k + Nx Ny)
    g = g - omega .* omega;

    Omega(:, pp) = omega;
    Theta(:, pp) = theta;
end

end