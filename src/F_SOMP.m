function [S, time] = F_SOMP(X, Y, p)
% transpose version of conventional SOMP

tic

Ny = size(Y, 1);
M  = size(Y, 2);
S = zeros(p, 1);
time = zeros(p, 1);
Q = zeros(p-1, M);

X = X ./ vecnorm(X, 2, 2);
if Ny > M
    Y = qr(Y, "econ"); % O(Ny M min(Ny, M)), Ref. Matrix Computation, p.278
    Y(diag(Y) < 1e-12, :) = [];
end

F = X * Y'; % O(N Ny M)
for pp = 1:p
    f = vecnorm(F, 2, 2); % O(N Ny)
    [~, Spp] = max(f);
    S(pp) = Spp;
    time(pp) = toc;

    if pp == p
        break
    end

    Qtmp = Q(1:pp-1, :);
    x = X(Spp, :);
    q = Orthonormal(x, Qtmp); % O(M k)
    F = F - (X * q') * (Y * q')'; % O(N M + Ny M + N Ny)
    Y = Y - (Y * q') * q; % O(Ny M)
    Q(pp, :) = q;
end

end

function q = Orthonormal(x, Q)

if isempty(Q)
    q = x;
else
    q = x - (x * Q') * Q; % O(M k)
end
q = q / norm(q);

end