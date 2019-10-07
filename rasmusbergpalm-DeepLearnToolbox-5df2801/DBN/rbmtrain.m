function rbm = rbmtrain(rbm, x, opts)
%rbm=dbn.rbm{1};x=train_x; 
%pre-train process
    assert(isfloat(x), 'x must be a float'); %判断一个函数是否成立，若不成立，则报错
    assert(all(x(:)>=0) && all(x(:)<=1), 'all data in x must be in [0:1]'); %检查所有元素在[0,1]之间
    m = size(x, 1); %样本数量
    numbatches = m / opts.batchsize; %样本组数  =m/100（每一组的样本数）—— 需要组数*每组的size==样本个数
    
    assert(rem(numbatches, 1) == 0, 'numbatches not integer');  %rem(x,y)求整除x/y的余数。

    for i = 1 : opts.numepochs     %opts.numepochs：迭代次数
        kk = randperm(m);  %产生1：m整数序列的一个随机排列
        err = 0;
        for l = 1 : numbatches
            batch = x(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :); %取出kk的第l组样本（实际上是随机取出的100个样本): 100*784
            
            %% CD-1计算过程 
            
            v1 = batch; %100*784
            h1 = sigmrnd(repmat(rbm.c', opts.batchsize, 1) + v1 * rbm.W'); %已知v1向上计算hidden层输出: repmat()函数将c'矩阵复制 batchsize行。100*100
                                                                           %Gibbs采样过程：对基于均匀分布采样的[0,1]区间的概率p >sigm函数所确定的概率时，该神经元输出为1，否则输出为0
            v2 = sigmrnd(repmat(rbm.b', opts.batchsize, 1) + h1 * rbm.W);  %已知hidden向下计算visiable层：100*784：Gibbs采样过程得到v2
            h2 = sigm(repmat(rbm.c', opts.batchsize, 1) + v2 * rbm.W');    %sigmod阈值函数，已知v2向上计算hidden层输出：100*100
            % Contrastive Divergence 的过程 
            c1 = h1' * v1;   %100*784
            c2 = h2' * v2;  %100*784
       %关于momentum，请参看Hinton的《A Practical Guide to Training Restricted Boltzmann Machines》  
       %它的作用是记录下以前的更新方向，并与现在的方向结合下，更有可能加快学习的速度  
            rbm.vW = rbm.momentum * rbm.vW + rbm.alpha * (c1 - c2)     / opts.batchsize; %（负）连接权增量
            rbm.vb = rbm.momentum * rbm.vb + rbm.alpha * sum(v1 - v2)' / opts.batchsize; %（负）u-1层偏置增量
            rbm.vc = rbm.momentum * rbm.vc + rbm.alpha * sum(h1 - h2)' / opts.batchsize; %（负）u层偏置增量

            rbm.W = rbm.W + rbm.vW; %连接权更新
            rbm.b = rbm.b + rbm.vb; %u-1层偏置更新
            rbm.c = rbm.c + rbm.vc; %u层偏置更新

            err = err + sum(sum((v1 - v2) .^ 2)) / opts.batchsize; %重构损失平方和均值：RBM的目标函数？？
        end
        
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Average reconstruction error is: ' num2str(err / numbatches)]);
        
    end
   
end
%dbn.rbm{1}=rbm;
