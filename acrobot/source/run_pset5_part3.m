% RUN THIS to generate your solution
megaclear

[p,xtraj,utraj,v,x0] = pset5_catch;

% if you want to display the trajectory again
%v.playback(xtraj);

% ********YOUR CODE HERE ********
% Set Q, R, and Qf for time varying LQR
% See problem statement for instructions here
xf = xtraj.eval(3);
q = xf(1:5);
qd = xf(6:10);
options = struct();
options.compute_gradients = true;
kinsol = p.doKinematics(q,qd,options);

% body index, so p.body(3) is the lower link
hand_body = 3;

% position of the "hand" on the lower link, 2.1m is the length
pos_on_hand_body = [0;-2.1];

% Calculate position of the hand in world coordinates
% the gradient, dHand_pos, is the derivative w.r.t. q
[hand_pos,dHand_pos,ddHand_pos] = p.forwardKin(kinsol,hand_body,pos_on_hand_body);

%Q = .05*eye(10);
Q = zeros(10);
R = .05;
Qf = zeros(10);
d2fdxb2 = 2;
d2fdzb2 = 2;
d2fdxbdtht1 = -2*dHand_pos(1,1); %\partial^2 f / \partial x_b \partial \theta_1
d2fdxbdtht2 = -2*dHand_pos(1,2);
d2fdzbdtht1 = -2*dHand_pos(2,1);
d2fdzbdtht2 = -2*dHand_pos(2,2);
d2fdtht12 = 2*(dHand_pos(1,1)^2+dHand_pos(2,1)^2);
d2fdtht1dtht2 = 2*(dHand_pos(1,1)*dHand_pos(1,2) + dHand_pos(2,1)*dHand_pos(2,2));
d2fdtht22 = 2*(dHand_pos(1,2)^2+dHand_pos(2,2)^2);
Qf(1:4,1:4) = [d2fdtht12 d2fdtht1dtht2 d2fdxbdtht1 d2fdzbdtht1
               d2fdtht1dtht2 d2fdtht22 d2fdxbdtht2 d2fdzbdtht2
               d2fdxbdtht1 d2fdxbdtht2 d2fdxb2 0
               d2fdzbdtht1 d2fdzbdtht2 0 d2fdzb2];
%Qf = 1e-6*Qf;
%Qf = 60*Qf;
% *******************************
options.sqrtmethod=false;
c = p.tvlqr(xtraj,utraj,Q,R,Qf,options);
sys_cl = p.feedback(c);

%%
% x0_test = x0;
% x0_test(3) = x0(3) + .1;
% traj_test_1 = sys_cl.simulate(xtraj.tspan,x0_test);
% v.drawWrapper(traj_test_1.tspan(2),traj_test_1.eval(traj_test_1.tspan(2)));

x0_test = x0 + .07*(rand(10,1) - 1);
traj_test_2 = sys_cl.simulate(xtraj.tspan,x0_test);
v.drawWrapper(traj_test_2.tspan(2),traj_test_2.eval(traj_test_2.tspan(2)));
playbackMovie(v,traj_test_2,'catch_perturb.avi');
% submit x_grade below
%x_grade = [traj_test_1.eval(xtraj.pp.breaks) traj_test_2.eval(xtraj.pp.breaks) Qf repmat(xtraj.tspan(2),10,1)]';

format short
%display(x_grade)
format long