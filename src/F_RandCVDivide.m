function [Itrain, Itest] = F_RandCVDivide(N, N_CV)

Itrain = cell(N_CV, 1);
Itest  = cell(N_CV, 1);

rng(0);
irandall = randperm(N);
ies = round((1:N_CV) / N_CV * N);
iss = [1, ies(1:N_CV-1) + 1];

for j = 1:N_CV
    Itrain{j} = irandall([1:iss(j)-1, ies(j)+1:N]);
    Itest {j} = irandall(iss(j):ies(j));
end

end