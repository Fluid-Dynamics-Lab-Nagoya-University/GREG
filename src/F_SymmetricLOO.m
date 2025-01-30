function [Itrain, Itest] = F_SymmetricLOO(N)

N_CV = ceil(N / 2);
Itrain = cell(N_CV, 1);
Itest  = cell(N_CV, 1);

iall = 1:N;
for j = 1:N_CV
    itest = [j, N + 1 - j];
    itrain = iall;
    itrain(itest) = [];
    Itrain{j} = itrain;
    Itest {j} = itest;
end

end