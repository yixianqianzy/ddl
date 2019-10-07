function rbm = rbmtrain(rbm, x, opts)
%rbm=dbn.rbm{1};x=train_x; 
%pre-train process
    assert(isfloat(x), 'x must be a float'); %�ж�һ�������Ƿ�����������������򱨴�
    assert(all(x(:)>=0) && all(x(:)<=1), 'all data in x must be in [0:1]'); %�������Ԫ����[0,1]֮��
    m = size(x, 1); %��������
    numbatches = m / opts.batchsize; %��������  =m/100��ÿһ��������������� ��Ҫ����*ÿ���size==��������
    
    assert(rem(numbatches, 1) == 0, 'numbatches not integer');  %rem(x,y)������x/y��������

    for i = 1 : opts.numepochs     %opts.numepochs����������
        kk = randperm(m);  %����1��m�������е�һ���������
        err = 0;
        for l = 1 : numbatches
            batch = x(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :); %ȡ��kk�ĵ�l��������ʵ���������ȡ����100������): 100*784
            
            %% CD-1������� 
            
            v1 = batch; %100*784
            h1 = sigmrnd(repmat(rbm.c', opts.batchsize, 1) + v1 * rbm.W'); %��֪v1���ϼ���hidden�����: repmat()������c'������ batchsize�С�100*100
                                                                           %Gibbs�������̣��Ի��ھ��ȷֲ�������[0,1]����ĸ���p >sigm������ȷ���ĸ���ʱ������Ԫ���Ϊ1���������Ϊ0
            v2 = sigmrnd(repmat(rbm.b', opts.batchsize, 1) + h1 * rbm.W);  %��֪hidden���¼���visiable�㣺100*784��Gibbs�������̵õ�v2
            h2 = sigm(repmat(rbm.c', opts.batchsize, 1) + v2 * rbm.W');    %sigmod��ֵ��������֪v2���ϼ���hidden�������100*100
            % Contrastive Divergence �Ĺ��� 
            c1 = h1' * v1;   %100*784
            c2 = h2' * v2;  %100*784
       %����momentum����ο�Hinton�ġ�A Practical Guide to Training Restricted Boltzmann Machines��  
       %���������Ǽ�¼����ǰ�ĸ��·��򣬲������ڵķ������£����п��ܼӿ�ѧϰ���ٶ�  
            rbm.vW = rbm.momentum * rbm.vW + rbm.alpha * (c1 - c2)     / opts.batchsize; %����������Ȩ����
            rbm.vb = rbm.momentum * rbm.vb + rbm.alpha * sum(v1 - v2)' / opts.batchsize; %������u-1��ƫ������
            rbm.vc = rbm.momentum * rbm.vc + rbm.alpha * sum(h1 - h2)' / opts.batchsize; %������u��ƫ������

            rbm.W = rbm.W + rbm.vW; %����Ȩ����
            rbm.b = rbm.b + rbm.vb; %u-1��ƫ�ø���
            rbm.c = rbm.c + rbm.vc; %u��ƫ�ø���

            err = err + sum(sum((v1 - v2) .^ 2)) / opts.batchsize; %�ع���ʧƽ���;�ֵ��RBM��Ŀ�꺯������
        end
        
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Average reconstruction error is: ' num2str(err / numbatches)]);
        
    end
   
end
%dbn.rbm{1}=rbm;
