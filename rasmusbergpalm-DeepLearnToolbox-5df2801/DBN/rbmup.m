function x = rbmup(rbm, x)
    x = sigm(repmat(rbm.c', size(x, 1), 1) + x * rbm.W'); %60000*100; 100为hidden结点个数——输出sigmod函数的值作为下一层RBM的输入。
end
