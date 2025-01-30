function [S, time, iR, det_test] = F_BDG(Ur, Sr, Un, Sn, p)

tic;

[n, r] = size(Ur);
Sn_sq = Sn .* Sn;
USsq = Un * Sn_sq; % O(N r_2)
t_vec = sum(USsq .* Un, 2); % O(N r_2)
Ck = zeros(0, r);
iR = [];
initial = true;
det_test = zeros(p, 1);
S = zeros(p, 1);
time = zeros(p, 1);
Ck_mem = zeros(p, r);
%%

for k = 1:p
    if initial == true   % initialize W&Cpp
        RC = iR * Ck;
        W = inv(Sr .* Sr);
        Winv = Sr .* Sr;
        initial = false;
    end
    %% searching        
    if k == 1
        s_vec = zeros(n, 0);
        iR = 0;
    else
        s_vec = USsq * (Un(S(1:k-1), :))'; % O(N r_2 k)
    end
    
    diff = s_vec * RC - Ur; % O(N r_1 k)
    nume = sum((diff * Winv) .* diff, 2); % O(N r_1^2)
    dnm = t_vec - sum((s_vec * iR) .* s_vec, 2); % O(N r_1 k)
    det_vec2 = nume ./ dnm;
    
    det_vec2(S(1:k-1)) = 0;
    [det_test(k), Sk] = max(det_vec2);   % argmaxdet
    S(k) = Sk;
    time(k) = toc;
    
%%   Update iR&C after we get pp-th sensor  
    s = zeros(1, k-1);
    u_i = Ur(Sk, :);
    
    for l = 1:(k-1)
        s(1, l) = Un(Sk, :) * Sn_sq * Un(S(l), :)';
        % O(M)
    end
    t = Un(Sk, :) * Sn_sq * Un(Sk, :)';
    
    diff = s * RC - u_i;
    dnm = t - s * iR * s';
    W = W + diff' * (dnm \ diff); % O(N r_1^2)
    Winv = inv(W);

    Ck_mem(k, :) = u_i;
    Ck = Ck_mem(1:k, :);
    
    sR = s * iR;
    iR_new = zeros(k, k);
    iR_new(1:k-1, 1:k-1) = iR;
    iR = iR_new + [sR'; -1] * (dnm \ [sR -1]); % O(k^2)
    RC = iR * Ck; % O(r_1 k^2)
end

end