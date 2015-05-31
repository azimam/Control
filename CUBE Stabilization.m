%%  Final Project-Lnear Systems and Control
%%  Stabilization of a CUBE system
%%  Fall 2012

clear all;
format long;
A = [   0             0           1          0         0;... 
        0             0           0          1         41.5436;...
        41.5436      -46.9409     0      -191.6471     -36.3892; ...
        -36.3892    -101.9403     0      -786.3512     0;... 
        0              1          0         0           0; ]
    
B = [   0;  0;   33.7528;   138.4917; 0 ];
 
C = [ 0  1  0    0   0 ;0   0   1   0   0 ];
 
D = [ 0 ;0 ];
 
%% Task 1:  the Model
Mc = [B,A*B,A*A*B,A*A*B,A*A*A*A*B];
controllability_rank = rank(Mc);
if ( controllability_rank == 5 )
    fprintf('\n The system is controllable \n');
elseif ( controllability_rank < 5 )
    fprintf('\n The system is not controllable \n');
end
 
% 2. Verification of the observability of the cube system
 
Mo = [C;C*A;C*A*A;C*A*A*A;C*A*A*A*A];
observability_rank = rank(Mo);
if ( observability_rank == 5 )
    fprintf('\n The system is observable \n');
elseif ( observability_rank < 5 )
    fprintf('\n The system is not observable \n');
end
 
%% Task 2: State-Feedback Control
 syms s;
alpha = sym2poly(det(s*eye(5) - A)); 
alpha = alpha(2:end);
Mb = eye(5);
for i = 1 : 5
    Mb(i,i+1:5) = alpha(1:end-i); 
end
Pinv = Mc*Mb;
P = inv(Pinv)
 
% Find kbar
eigen = [-11 -12 -13 -14 -15];
eigenpoly = 1;
for i = 1 : length(eigen)
   eigenpoly = eigenpoly*(s-eigen(i));
end
beta = sym2poly(eigenpoly);
beta = beta(2:end);
kbar = beta - alpha
 
% Find K
K   = kbar*P

%% Task 3:  Estimator % d(xhat)/dt = (A-L*C)*xhat + L*y + B*U
L = place(A',C',eigen)'; 
Ahat = A-L*C; 
Bhat = [L B] 
