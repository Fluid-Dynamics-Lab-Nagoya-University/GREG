function [S, time] = F_REG(X, p)
    
    tic

    N = size(X, 1);
    S = zeros(p, 1);
    Omega = zeros(N, p-1);
    time = zeros(p, 1);

    R = qr(X, "econ"); % O(N M^2)
    f = vecnorm(X * R', 2, 2) .^ 2; % O(N M^2)
    g = vecnorm(X, 2, 2) .^ 2; % O(N M)
    
    Is = (1:N)';
    S_pp = [];
    for pp = 1:p
        Is(g(Is) < 1e-10) = [];
        Is = setdiff(Is, S_pp);
        v = f(Is) ./ g(Is);
%         v(g < 1e-10) = 0;
        [~, T_pp] = max(v);
        S_pp = Is(T_pp);
        S(pp) = S_pp;
        time(pp) = toc;

        Omega_tmp = Omega(:, 1:pp-1);
        delta = X * X(S_pp, :)' - Omega_tmp * Omega_tmp(S_pp, :)'; % O(N M)
        omega = delta / sqrt(delta(S_pp));
        Omega(:, pp) = omega;

        f = f + omega .* (-2 * X * (X' * omega) ...
            + 2 * Omega_tmp * (Omega_tmp' * omega) + norm(omega)^2 * omega); % O(N M + N k)
        g = g - omega .* omega;
    end


end