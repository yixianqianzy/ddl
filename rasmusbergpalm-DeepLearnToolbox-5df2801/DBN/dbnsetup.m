function dbn = dbnsetup(dbn, x, opts)
    n = size(x, 2);
    dbn.sizes = [n, dbn.sizes];  %���������ݵ�ά�ȼ���DBN size�ĵ�һ��ά�� 784*100

    for u = 1 : numel(dbn.sizes) - 1   %����dbn.sizes�е�Ԫ�ظ���-1
        dbn.rbm{u}.alpha    = opts.alpha;
        dbn.rbm{u}.momentum = opts.momentum;

        dbn.rbm{u}.W  = zeros(dbn.sizes(u + 1), dbn.sizes(u)); %��ʼ�����ز�u+1��u��֮�������Ȩ��Ϊ0���� 100*784
        dbn.rbm{u}.vW = zeros(dbn.sizes(u + 1), dbn.sizes(u)); %��ʼ��Ȩ������Ϊ0���� 100*784

        dbn.rbm{u}.b  = zeros(dbn.sizes(u), 1);  %��ʼ����u���ƫ�þ���Ϊ0�� 784*1
        dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);  %��ʼ��ƫ������Ϊ0��784*1

        dbn.rbm{u}.c  = zeros(dbn.sizes(u + 1), 1); %��ʼ����u+1���ƫ�þ���Ϊ0��100*1
        dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1); %��ʼ��ƫ������Ϊ0��100*1
    end

end
