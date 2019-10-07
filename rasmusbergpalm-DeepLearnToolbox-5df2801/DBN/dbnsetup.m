function dbn = dbnsetup(dbn, x, opts)
    n = size(x, 2);
    dbn.sizes = [n, dbn.sizes];  %将输入数据的维度加在DBN size的第一个维度 784*100

    for u = 1 : numel(dbn.sizes) - 1   %计算dbn.sizes中的元素个数-1
        dbn.rbm{u}.alpha    = opts.alpha;
        dbn.rbm{u}.momentum = opts.momentum;

        dbn.rbm{u}.W  = zeros(dbn.sizes(u + 1), dbn.sizes(u)); %初始化隐藏层u+1与u层之间的链接权重为0矩阵 100*784
        dbn.rbm{u}.vW = zeros(dbn.sizes(u + 1), dbn.sizes(u)); %初始化权重增量为0矩阵 100*784

        dbn.rbm{u}.b  = zeros(dbn.sizes(u), 1);  %初始化第u层的偏置矩阵为0； 784*1
        dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);  %初始化偏置增量为0；784*1

        dbn.rbm{u}.c  = zeros(dbn.sizes(u + 1), 1); %初始化第u+1层的偏置矩阵为0；100*1
        dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1); %初始化偏置增量为0；100*1
    end

end
