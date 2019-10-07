function X = sigmrnd(P)
    % X = double(1./(1+exp(-P)))+1*randn(size(P));
    X = double(1./(1+exp(-P)) > rand(size(P))); %rand（）产生[0,1]之间的随机数：gibbs采样过程
end