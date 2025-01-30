function [S, time] = F_DG(U, p)

r = size(U, 2);
time = zeros(p, 1);
S = zeros(p, 1);

[Sr, time1] = DG_r(U);
min_rp = min([r; p]);
S(1:min_rp) = Sr(1:min_rp);
time(1:min_rp) = time1;

if p > r
    [S, time2] = DG_p(U, S);
    time(r+1:end) = time1 + time2;
end

end

% See Algorithm 1 in [Determinant-Based Gast Greedy Sensor Selection
% Algorithm]
function [S, time] = DG_r(U)

tic;
[~, ~, S] = qr(U', "econ", "vector");
time = toc;    

end

% See Algorithm 2 in [Determinant-Based Gast Greedy Sensor Selection
% Algorithm]
function [S, time] = DG_p(U, S)

[n, r] = size(U);
p = length(S);

time = zeros(p-r, 1);
tic;

C   = U(S(1:r), :);
CTC = C' * C;
CTCI = inv(CTC);
J = zeros(n, 1);

for pp = r+1:p
    for nn = 1:n
        v = U(nn, :);
        J(nn) = 1 + v * CTCI * v';
    end
    J(S(1:pp-1)) = 0;
    [~, Spp] = max(J);
    time(pp - r) = toc;

    v = U(Spp, :);
    CTCI = CTCI * (eye(r) - 1 / J(Spp) * (v' * v) * CTCI);
    S(pp) = Spp;
end

end