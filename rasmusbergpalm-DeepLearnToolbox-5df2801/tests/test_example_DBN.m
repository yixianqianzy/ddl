%function test_example_DBN
load mnist_uint8;

train_x = double(train_x) / 255; %The original data is within [0,255], the opeartion constraint each data within [0,1]; Train_x(dim)=60000*784
train_y = double(train_y);  %each data within in {0,1}; Train_y=60000*10

test_x  = double(test_x)  / 255; %Test_x(dim)=10000*784
test_y  = double(test_y); %Test_y=10000*10

%%  ex1 train a 100 hidden unit RBM and visualize its weights
rand('state',0)   % �����������������䣺ÿ�β��������֮ǰ�������������ܲ�����ͬ���������
dbn.sizes = [4]; %4��DBN
opts.numepochs =   1;   %��������=1
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
dbn = dbnsetup(dbn, train_x, opts); %����DBN�Ĳ��������Բ��ֲ�����ʼ��
dbn = dbntrain(dbn, train_x, opts); %Pre-train DBN����DBN�ֳ����ɸ�RBM���ֱ��ÿ��RBMѧϰ��Ȩ�ؾ����ƫ��

figure; visualize(dbn.rbm{1}.W'); 

%unfold dbn to nn: ��dbnչ����nn
nn = dbnunfoldtonn(dbn, 10);
nn.activation_function = 'sigm';

%train nn
opts.numepochs =  1;  %numepochs��ѭ���Ĵ���  
opts.batchsize = 100; %ÿ����������
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);

assert(er < 0.10, 'Too big error');
