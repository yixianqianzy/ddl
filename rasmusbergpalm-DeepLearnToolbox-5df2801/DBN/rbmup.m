function x = rbmup(rbm, x)
    x = sigm(repmat(rbm.c', size(x, 1), 1) + x * rbm.W'); %60000*100; 100Ϊhidden�������������sigmod������ֵ��Ϊ��һ��RBM�����롣
end
