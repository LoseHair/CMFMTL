function [predict_p,predict_n]=CMFMTL(A_p,A_n,drug_sim,sim_dis,alpha,beta,lam,dimension_of_latent_vector) 
[M,N]=size(A_p);
k=dimension_of_latent_vector;
rng(0,'twister');
U=rand(M,k);
V=rand(N,k);
R_p=rand(k,k);
R_n=rand(k,k);

D_U=diag(sum(drug_sim,2));
D_V=diag(sum(sim_dis,2));
L_U=D_U-drug_sim;
L_V=D_V-sim_dis;
rho_1=1;
rho_2=1;
Z=0;
Y=0;

max_iter=500;
error=0.001;
for i=1:max_iter
    SE_old=U*R_p*V'+U*R_n*V';
    
    W=((alpha*L_U+rho_1*eye(M))^-1)*(rho_1*U-Z);
    J=((beta*L_V+rho_2*eye(N))^-1)*(rho_2*V-Y);
    
    R_p=CG(0,U'*U,V'*V,U'*A_p*V,lam,0.1);
    R_n=CG(0,U'*U,V'*V,U'*A_n*V,lam,1);
    
    U=(A_p*V*R_p'+A_n*V*R_n'+Z+rho_1*W)*((R_p*V'*V*R_p'+R_n*V'*V*R_n'+(rho_1+lam)*eye(k))^-1);
    V=(A_p'*U*R_p+A_n'*U*R_n+Y+rho_2*J)*(((U*R_p)'*U*R_p+(U*R_n)'*U*R_n+(rho_2+lam)*eye(k))^-1);
    Y=Y+rho_2*(J-V);
    Z=Z+rho_1*(W-U);
    rho_2=1.1*rho_2;
    rho_1=1.1*rho_1; 
   
    SE_new=U*R_p*V'+U*R_n*V';
    e=norm(SE_new-SE_old,'fro')/norm(SE_old,'fro');
    
    if e<0.0000001
        i
        break;
    end
end
predict_p=U*R_p*V';
predict_n=U*R_n*V';

end


function Y=CG(X_in,A,B,D,mu,tol)

X=X_in;
R=D-A*X*B-mu*X;
P=R;

for i=1:50
    R_norm=trace(R*R');
    Q=A*P*B+mu*P;
    alpha=R_norm/(trace(Q*P'));
    X=X+alpha*P;
    R=R-alpha*Q;
    err=norm(R,'fro');
    if err<tol
%         i
        break
    end
    beta=trace(R*R')/R_norm;
    P=R+beta*P;
end
Y=X;
end