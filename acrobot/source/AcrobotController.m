classdef AcrobotController < DrakeSystem
  properties
    p
    closer
  end
  methods
    function obj = AcrobotController(plant)
      obj = obj@DrakeSystem(0,0,4,1,true,true);
      obj.p = plant;
      obj.closer = 0;
      obj = obj.setInputFrame(plant.getStateFrame);
      obj = obj.setOutputFrame(plant.getInputFrame);
    end
    
    function u = output(obj,t,~,x)
      q = x(1:2);
      qd = x(3:4);
      
      % unwrap angles q(1) to [0,2pi] and q(2) to [-pi,pi]
      q(1) = q(1) - 2*pi*floor(q(1)/(2*pi));
      q(2) = q(2) - 2*pi*floor((q(2) + pi)/(2*pi));

      %%%% put your controller here %%%%
      % You might find some of the following functions useful
      % user definitions (for first part of problem, set k2=k3=0)
      % k1 = 6 shows that energy-shaping control leads to nearly constant
      % energy when looking at position relative to lower (stable)
      % equilibrium
      k1 = 6; % energy-shaping gain
      k2 = 6; % partial feedback linearization position gain
      k3 = 6; % partial feedback linearization rate gain
      tol = 500; % cost threshold for switching to LQR feedback controller
      firstPart = 0; % set to 1 when looking at energy-shaping only controller, otherwise set 0 (false)
      
      % error vector
      if firstPart
        e  = [q;qd] - zeros(4,1); % for first part of problem (stable equilibrium)
      else
        e = [q;qd]-[pi;0;0;0]; % for second part of problem (unstable
      % equilibrium)
      end
      
      % get dynamics parameters
      [H,C,B] = obj.p.manipulatorDynamics(q,qd);
      [f,df] = obj.p.dynamics(t,[pi;0;0;0],0);
      
      % construct LQR 
      Alin = df(:,2:5);
      Blin = df(:,6);
      Q = .5*diag([1 1 1 1]);
      R = .5;
      [K,S] = lqr(Alin,Blin,Q,R);
      
      % energy-shaping piece
      com_position = obj.p.getCOM(q); % center-of-mass position
      mass = obj.p.getMass();
      gravity = obj.p.gravity;
      % Recall that the kinetic energy for a manipulator given by .5*qd'*H*qd
      T = 1/2*qd'*H*qd;
      U = -mass*gravity(3)*com_position(2);
      E = T+U;
      com_position_d = obj.p.getCOM([pi;0]);
      Ed = -mass*gravity(3)*com_position_d(2);
      ue = k1*(Ed - E)*qd(2);
      
      % partial feedback linearization piece
      Hinv = pinv(H);
      a2 = Hinv(1,2); % =H(2,1)
      a3 = Hinv(2,2);
      y  = -k2*q(2)-k3*qd(2);
      up = (y+a2*C(1))/a3 + C(2);
      
      % if the cost (e'*S*e) is smaller than some tolerance for
      % the first time, set the closer flag so that the control input
      % should be from LQR
      if (e'*S*e < tol && ~obj.closer && ~firstPart)
          obj.closer = 1;
      end
      
      % if we haven't gotten to within the linearization regime,
      % our control input should be the sum of partial feedback
      % linearization and energy-shaping inputs
      if (~obj.closer)
          u  = up+ue;
      % otherwise, if we are in a region where the linearization
      % is valid, use LQR gain applied to error vector
      else
          u = -K*e;
          %u = 0;
      end
      %%%% end of your controller %%%%
      
      % leave this line below, it limits the control input to [-20,20]
      u = max(min(u,20),-20);
      % This is the end of the function
    end
  end
  
  methods (Static)
    function [t,x,x_grade]=run()
      plant = PlanarRigidBodyManipulator('Acrobot.urdf');
      controller = AcrobotController(plant);
      v = plant.constructVisualizer;
      sys_closedloop = feedback(plant,controller);
      
      x0 = [.1*(rand(4,1) - 1)]; % start near the downward position
      %x0 = [pi - .1*randn;0;0;0];  % start near the upright position
      xtraj=simulate(sys_closedloop,[0 10],x0);
      v.axis = [-4 4 -4 4];
      playback(v,xtraj);
      %playbackMovie(v,xtraj,'swingup_cloop_noLQR.avi');
      t = xtraj.pp.breaks;
      x = xtraj.eval(t);
      t_grade = linspace(3,4,98);
      x_grade = [xtraj.eval(0) xtraj.eval(t_grade) xtraj.eval(10)];
      x_grade = x_grade';
    end
  end
end
