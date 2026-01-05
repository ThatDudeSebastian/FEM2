% ========================================================================
% Finite Element Code for trusses
% ========================================================================

% tic
clf;
clear variables;
format short;

%% Pre-processing
truss_triangle_input;

% Kronecker delta
I = eye(ndm,ndm);

%% FE-Analysis

%initialize displacement list
u    = zeros(nnp*ndf,1);

% time step / load step variables
time = 0;       % initialise current time
nsteps = floor(ttime/dt) + 1;

% newton iteration variables
tol = 1e-6;
maxiter = 20;

%Initialisation of elem array
AA = struct('sig',zeros(nqp,1));
elem = repmat(AA,nel,1)

E = xE;

for step = 1:nsteps
  fprintf(1,'time = %8.4e\n',time)  
  
  % load boundary conditions for current timestep
  % set up prescribed displacement vector from Dirichlet bc's
  u_d = zeros(numDrltDofs,1);
  
  for i=1:numDrltDofs
    node  = drlt(i,1);
    ldof  = drlt(i,2);
  # loadid = drlt(i,3);
  # scale = drlt(i,4);
    drltDofs(i) = (node-1)*ndf + ldof;
  %%% Optional: account for prescribed movements (let's ignore that for now)
  #  u_d(i) = scale * interp1(loadcurve(loadid).time, loadcurve(loadid).value, time);
  end
  
  freeDofs = setdiff(allDofs, drltDofs);
  
  % set up prescribed force vector from Neumann bc's
  fpre = zeros(nnp*ndf,1);
  for i=1:size(neum, 1);
    node  = neum(i,1);
    ldof  = neum(i,2);
    loadid = neum(i,3);
    scale = neum(i,4);
    
    fpre((node-1)*ndf+ldof) = scale * interp1(loadcurve(loadid).time, loadcurve(loadid).value, time);
  end
  
  %initialise rsn to guarantee one excecution
  rsn = 1;
  iter = 0;
  
  while rsn > tol
        
    % Initialisation of global vectors and matrix
    K    = zeros(nnp*ndf,nnp*ndf);
    fext = zeros(nnp*ndf,1);
    fint = zeros(nnp*ndf,1);
    fvol = zeros(nnp*ndf,1);
    frea = zeros(nnp*ndf,1);    
    
    
    % Loop over the number of elements, Summation over all elements 'e'
    for e=1:nel
      
      %     fprintf('Element, e = %i\n', e);
      
      % Coordinates of the element nodes
      % Hint : to be extracted from the global coordinate list 'x'
      %        considering the 'conn' matrix
      xe = zeros(ndm,nen);
      for i = 1:ndm
        xe(i,:) = x(conn(e,:),i)';
      end
      
      % Call the coordinates and weights of the gauss points
      % Hint : input parameters - nqp
      %        output parameters - xi, w8
      %        function name - gauss2d
      [xi,w8] = gauss(nqp,ndm);
      
      %calculate gdof
      gdof = zeros(nen*ndf,1);
      for node=1:nen
        gdof(node*ndf-1:node*ndf) = [conn(e,node)*ndf-1 : conn(e,node)*ndf]'; 
      end
      
      %initialise Ke fvole
      Ke = zeros(nen*ndf,nen*ndf);
      fvole = zeros(nen*ndf,1);
      finte = zeros(nen*ndf,1);
      
      % Loop over the gauss points, Summation over all gauss points 'q'
      for q = 1:nqp
        
        % Call the shape functions and its derivatives
        % Hint : input parameters - xi(q), nen
        %        output parameters - N, gamma
        %        function name - shape2d
        [N,gamma] = shape(xi(q),nen,ndm);
        
        % Gmatrix for coordinate rotation                          
        l=sqrt((xe(1,2)-xe(1,1))^2+(xe(2,2)-xe(2,1))^2);
        cosphi= (xe(1,2) - xe(1,1))/l;
        sinphi= (xe(2,2) - xe(2,1))/l;
        G_u= [cosphi , sinphi, 0, 0];
        G_l= [0, 0, cosphi, sinphi];
        g_A = gamma(1);
        g_B = gamma(2);
        G_matrix = [G_u; G_l];
        
        % Determinant of Jacobian and the inverse of Jacobian
        % at the quadrature points q
        % Hint : For this 1d-case the Jacobian is a scalar 
        xe_loc = [0 l];
        [detJq,invJq] = jacobian(xe_loc,gamma,nen,ndm);
        
        %displacements along the truss axis
        ue = zeros(ndf, nen);
        for i=1:nen
          gi = conn(e,i);
          ue(:,i) = u(gi*ndf-1 : gi*ndf);
        end
        ue_vec = reshape(ue,4,1);
        ue_loc = G_matrix*ue_vec;
        
        %Calculate strain in the truss element
        G = gamma*invJq;
        eps = ue_loc'*G;        %displacement gradient in current quad.point
        
        
        %material model
        %Stress in current quadrature point: 1d Hooke's law
        sig = E * eps;    
        C = E;        
        
        %internal forces
        finte = finte + (g_A*G_u+g_B*G_l)' * Area * sig *  w8(q); 
        
        %stiffness matrix
        Ke = Ke + (g_A*G_u+g_B*G_l)'*C*Area*(g_A*G_u+g_B*G_l)*invJq*w8(q);
        
        %save stresses
        elem(e).sig(q) = sig;        
        
        
      end % Loop over Gauss points
      %assemble element matrices into global matrices
      K(gdof,gdof) = K(gdof,gdof) + Ke;
      fvol(gdof) = fvol(gdof) + fvole;
      fint(gdof) = fint(gdof) + finte;
    end % Loop over elements

    %% Newton - Raphson solution step
    % calculate residuum of free DOF 's
    fext = fpre + fvol;
    rsd_F = #TODO % see equation (8.1)
    
    % calculate the extended residuum norm 
    % extension garantuees update in first iteration for changed BC's
    rsn = norm(rsd_F) + norm(u_d - u(drltDofs));
    
    fprintf(1, ' %2d. residuum norm= %e\n', iter, rsn);
    
    %check whether an update has to be performed (only if rsn > tol)
    if # TODO
      %calculate the right hand side
      if iter == 0 
        % apply increment of drlt-BC's in the first iteration
        rhs = #TODO
        u(drltDofs) = #TODO
      else
        rhs = #TODO
      end
      
      %perform the newton-update
      du = zeros(size(u));
      du(freeDofs) = #TODO %see equation (8.6)
      u(freeDofs) = #TODO	%see equation (8.7)
      eps_v = eps_v_bu;
    end
    
    % raise the iter-counter and check whether maxiter is reached
    iter = #TODO
    if #TODO
      error('no convergence in newton scheme')
    end
    
  end %Newton loop
  
  % Compute the total force vector
  % fprintf('Solution : Total force vector\n');
  fext(freeDofs) = fvol(freeDofs) + fpre(freeDofs);
  fext(drltDofs) = K(drltDofs,freeDofs)*u(freeDofs) + K(drltDofs,drltDofs)*u(drltDofs);
  
  % Compute the reaction forces at the Dirichlet boundary
  % fprintf('Solution : Reaction force vector\n\n');
  frea(drltDofs) = fext(drltDofs)- fvol(drltDofs);
  % frea;
  
  xplt = x+reshape(u, 2, [])';
  
  hold off
  for i = 1:size(conn, 1)
    plot (x([conn(i,1), conn(i,2)],1), x([conn(i,1), conn(i,2)],2), 'linewidth',1, 'k-')
    plot (xplt([conn(i,1), conn(i,2)],1), xplt([conn(i,1), conn(i,2)],2), 'linewidth', 2, 'g-')
    axis equal
    hold on
  endfor

  ylim([-6, 1])
  xlim([0,13])
  axis equal
  hold on
  drawnow
  
  
  % % Update time
  time = #TODO
  pause(0.05)
  
end %loop over time steps / load steps
