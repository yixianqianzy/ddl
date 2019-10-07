function dbn = dbntrain(dbn, x, opts)
    n = numel(dbn.rbm);

    dbn.rbm{1} = rbmtrain(dbn.rbm{1}, x, opts); %ѵ����һ��RBM�Ĳ���
    for i = 2 : n
        x = rbmup(dbn.rbm{i - 1}, x);  %����ڶ���RBM������
        dbn.rbm{i} = rbmtrain(dbn.rbm{i}, x, opts); %ѵ���ڶ���RBM�Ĳ���
    end

end
