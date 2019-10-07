function L = rating_loss(Train, B, D, r)
I = Train==0;
pre_mat = B'*D;
pre_mat(I)=0;
dif_mat = 2*r*Train - r - pre_mat;
L = norm(dif_mat,'fro')^2;
end