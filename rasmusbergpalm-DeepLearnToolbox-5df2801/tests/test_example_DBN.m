%function test_example_DBN
load mnist_uint8;

train_x = double(train_x) / 255; %The original data is within [0,255], the opeartion constraint each data within [0,1]; Train_x(dim)=60000*784
train_y = double(train_y);  %each data within in {0,1}; Train_y=60000*10

test_x  = double(test_x)  / 255; %Test_x(dim)=10000*784
test_y  = double(test_y); %Test_y=10000*10

%%  ex1 train a 100 hidden unit RBM and visualize its weights
rand('state',0)   % 控制随机数产生的语句：每次产生随机数之前添加上这个，就能产生相同的随机数。
dbn.sizes = [4]; %4层DBN
opts.numepochs =   1;   %迭代次数=1
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);
figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
%rand('state',1)
%train dbn
dbn.sizes = [100 100];
opts.numepochs =   1;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, train_x, opts); %设这DBN的参数，并对部分参数初始化
dbn = dbntrain(dbn, train_x, opts); %Pre-train DBN，将DBN分成若干个RBM，分别对每个RBM学习出权重矩阵和偏置

figure; visualize(dbn.rbm{1}.W'); 

%unfold dbn to nn: 将dbn展开成nn
nn = dbnunfoldtonn(dbn, 10);
nn.activation_function = 'sigm';

%train nn
opts.numepochs =  1;  %numepochs是循环的次数  
opts.batchsize = 100; %每组样本个数
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);

assert(er < 0.10, 'Too big error');
