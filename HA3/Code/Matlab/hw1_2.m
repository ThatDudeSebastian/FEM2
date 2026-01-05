%% Aufgabe 1 (Modal) + Aufgabe 2 (Transient)
% 2D Truss Model of Crane + Pick-Up Truck (4.1 t)
% Newmark-Beta Time Integration

clear; 
clc;
clf;
format short e;

%%  PART 1: INPUT DATA (Geometry, Material, BCs)

% 1. Problem Definition
ndm = 2;        % Spatial dimension
ndf = 2;        % Dofs per node (ux, uy)
nen = 2;        % Nodes per element

% 2. Geometry (Nodes and Connectivity)
conn = [
    1 8;  2 8;  2 3;  3 8;  3 9;  3 4;  4 9;  4 10; 4 5;  5 10; 
    5 11; 5 6;  6 11; 6 12; 6 7;  7 12; 8 9;  9 10; 10 11; 11 12; 
    7 13; 13 14; 13 15; 14 15; 14 16; 15 17;
];
nel = size(conn, 1);   % Number of elements

% Node coordinates [x y]
x = [ 0   0;
      1   0;
      2   0;
      4   0;
      6   0;
      8   0;
      9   0;
      1   0.4;
      3   0.4;
      5   0.4;
      7   0.4;
      8   0.4;
      9  -1;
      6  -2;
     12  -2;
      6  -4.5;
     12  -4.5 ];

nnp = size(x, 1);      % Number of nodes
ndof = nnp*ndf;
nqp = 1;               % 1 Gauss point

% 3. Material (S960QL)
xE   = 210e9;   % Young's modulus [Pa]
xRho = 7850;    % Density [kg/m^3]

% Cross-section
% "Für alle Stäbe die doppelte Querschnittsfläche eingeben"
% Assuming a base rod diameter of e.g. 7cm (0.07m) from footnote? 
% Let's use a reasonable structural area. 
r = 0.035; % Radius 3.5cm
base_area = pi * r^2; 
Area = 2 * base_area; %  doppelte Querschnittsfl¨ache

% 4. Truck Mass
truck_mass_total = 4100;      % [kg]
truck_nodes      = [16, 17]; 
mass_per_node    = truck_mass_total / numel(truck_nodes); % 2050 kg

% 5. Boundary Conditions (fixed)
% Dirichlet boundary condition
% [node, dof(1=ux,2=uy), loadid, scale]
drlt = [
    1, 1, 0, 0;
    1, 2, 0, 0;
    2, 1, 0, 0;
    2, 2, 0, 0;
];

% BC variables 
allDofs = (1:1:nnp*ndf)';       % array with all DOF numbers
numDrltDofs = size(drlt,1);     % number of dirichlet DOF's
drltDofs = zeros(numDrltDofs,1);% initialise drltDofs arrayf
for i = 1:numDrltDofs
    node = drlt(i,1);
    ldof = drlt(i,2);
    drltDofs(i) = (node-1)*ndf + ldof;
end

freeDofs = setdiff(allDofs, drltDofs); % difference set (all - drlt)


%%  PART 2: ASSEMBLY OF GLOBAL K AND M (STRUCTURE + TRUCK MASS)
%initialize
K = zeros(ndof, ndof);
M = zeros(ndof, ndof);    % structure mass + later truck mass

%fprintf('Assembling global stiffness K and mass M...\n');

%  1d trusses in 2d space

for e = 1:nel
    % Element node coordinates
    xe = x(conn(e,:), :)';      % 2x2: [x1 x2; y1 y2]
    dx = xe(1,2) - xe(1,1);
    dy = xe(2,2) - xe(2,1);
    L  = sqrt(dx^2 + dy^2);
    
    % Direction cosines
    c = dx / L;
    s = dy / L;
    
    % Element stiffness (Ke)
    [xi, w8] = gauss(nqp, ndm);
    Ke = zeros(nen*ndf, nen*ndf);
    
    for q = 1:nqp
        [N, gamma] = shape(xi(q), nen, ndm);
        xe_loc     = [0 L];
        [detJq, invJq] = jacobian(xe_loc, gamma, nen, ndm);
        
        G_u = [c, s, 0, 0];
        G_l = [0, 0, c, s];
        g_A = gamma(1);
        g_B = gamma(2);
        % B Matrix (strain-displacement in global DOFs)
       
        B_mat = (g_A*G_u + g_B*G_l);   % 1x4

        % Stiffness Accumulation (1D bar: EA * B^T * B * (1/J) * w)
       
        Ke = Ke + B_mat' * xE * Area * B_mat * invJq * w8(q);
    end
    
    % Element mass (lumped)
    m_elem  = xRho * Area * L;
    Me_diag = [m_elem/2, m_elem/2, m_elem/2, m_elem/2];
    
    % Global DOF indices for this element
    gdof = zeros(nen*ndf, 1);
    for node = 1:nen
        gdof(node*ndf-1 : node*ndf) = ...
            (conn(e,node)*ndf-1 : conn(e,node)*ndf);
    end
    
    % Assemble K
    K(gdof, gdof) = K(gdof, gdof) + Ke;
    
    % Assemble M (structure part)
    for i = 1:length(gdof)
        dof_idx = gdof(i);
        M(dof_idx, dof_idx) = M(dof_idx, dof_idx) + Me_diag(i);
    end
end

% Add truck point mass to M
fprintf('Adding truck mass (%.1f kg) to nodes %s...\n', ...
        truck_mass_total, mat2str(truck_nodes));
for n_idx = 1:numel(truck_nodes)
    node_num = truck_nodes(n_idx);
    dof_x    = (node_num-1)*ndf + 1;
    dof_y    = (node_num-1)*ndf + 2;
    
    M(dof_x, dof_x) = M(dof_x, dof_x) + mass_per_node;
    M(dof_y, dof_y) = M(dof_y, dof_y) + mass_per_node;
end


%%  PART 3: AUFGABE 1 – MODAL ANALYSIS (Eigenfrequencies + Mode Shapes)

fprintf('\n=== AUFGABE 1: Modal Analysis (with truck mass in M) ===\n');

K_red = K(freeDofs, freeDofs);
M_red = M(freeDofs, freeDofs);

num_modes_requested = 6;

[Phi_all, D_all] = eig(full(K_red), full(M_red));
eigenvalues      = diag(D_all);

% Clean tiny negative eigenvalues (numerical noise)
tol_clean = 1e-8;
eigenvalues(real(eigenvalues) < 0 & real(eigenvalues) > -tol_clean) = 0;

% order (kleinste ist erst, 1. mode kommt erst)
[vals, idx] = sort(real(eigenvalues));

tol_pos = 1e-6; % nur eigenvalue in plus  (rigid-body / mechanism entfernt)
posIdx  = find(vals > tol_pos);

if isempty(posIdx)
    error('No positive eigenvalues found. Check model or BCs.');
end

%wie viel modes aktuell verwendet werden
num_modes = min(num_modes_requested, numel(posIdx));
vals_use  = vals(posIdx(1:num_modes));
Phi_red   = Phi_all(:, idx(posIdx(1:num_modes)));

% eigen freqs
omegas = sqrt(vals_use);        % rad/s
freqs  = omegas / (2*pi);       % Hz

fprintf('======================================\n');
fprintf(' First %d positive eigenfrequencies\n', num_modes);
fprintf('======================================\n');
fprintf(' Mode |  Freq [Hz]  | Omega [rad/s]\n');
fprintf('--------------------------------------\n');
for i = 1:num_modes
    fprintf('  %d   |  %8.4f   |  %8.4f\n', ...
        i, freqs(i), omegas(i));
end
fprintf('--------------------------------------\n');

% Reconstruct full eigenvectors
Phi_full = zeros(ndof, num_modes);
Phi_full(freeDofs, :) = Phi_red;

% Plot mode shapes
figure(1);
clf;
set(gcf, 'Color', 'w');

plot_cols = 3;
plot_rows = 2;

for mode_idx = 1:num_modes
    subplot(plot_rows, plot_cols, mode_idx);
    hold off;
    
    % Undeformed shape
    for i = 1:nel
        plot(x([conn(i,1), conn(i,2)],1), ...
             x([conn(i,1), conn(i,2)],2), ...
             'k:', 'linewidth', 0.5);
        hold on;
    end
    
    % Deformed mode shape (scaled)
    u_mode = Phi_full(:, mode_idx);
    max_val = max(abs(u_mode));
    if max_val == 0
        scale = 1.0;
    else
        scale = 1.0 / max_val;
    end
    x_def = x + reshape(u_mode, 2, [])' * scale;
    
    for i = 1:nel
        plot(x_def([conn(i,1), conn(i,2)],1), ...
             x_def([conn(i,1), conn(i,2)],2), ...
             'b-', 'linewidth', 1.5);
    end
    
    % Truck nodes
    plot(x_def(truck_nodes,1), x_def(truck_nodes,2), ...
         'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 5);
    
    axis equal;
    title(sprintf('Mode %d: %.2f Hz', mode_idx, freqs(mode_idx)), ...
          'FontSize', 9);
    xlim([-2, 14]);
    ylim([-8, 2]);
    box on;
end


%%  PART 4: AUFGABE 2 – TRANSIENT ANALYSIS (Newmark-Beta)

fprintf('\n=== AUFGABE 2: Transient Response (Newmark-Beta) ===\n');

% Time settings
dt      = 0.01;    % time step [s]
T_total = 4.0;     % total time [s]
n_steps = floor(T_total/dt);

% Newmark parameters (average acceleration)
beta  = 0.25;
gamma = 0.50;

% Integration constants
a0 = 1 / (beta * dt^2);
a2 = 1 / (beta * dt);
a3 = 1 / (2*beta) - 1;
a6 = dt * (1 - gamma);
a7 = gamma * dt;

% External step load: truck weight on nodes 16 & 17 (downwards)
g = 9.81;
F_ext = zeros(ndof, 1);
for n_idx = 1:numel(truck_nodes)
    node_num = truck_nodes(n_idx);
    dof_y    = (node_num-1)*ndf + 2;
    F_ext(dof_y) = F_ext(dof_y) - mass_per_node * g;
end
F_red = F_ext(freeDofs);

% Initial conditions (at rest)
u_curr = zeros(ndof, 1);
v_curr = zeros(ndof, 1);
a_curr = zeros(ndof, 1);

% Initial acceleration: M_red * a0 = F_red - K_red*u0, u0=0
a_curr(freeDofs) = M_red \ F_red;

% Effective stiffness (undamped)
K_eff = K_red + a0 * M_red;
[L_eff, U_eff] = lu(K_eff);

% Storage for history
time_hist     = zeros(n_steps, 1);
disp_tip_hist = zeros(n_steps, 1);

% Track node 17 vertical displacement
track_node = 17;
track_dof  = (track_node-1)*ndf + 2;

fprintf('Starting transient simulation (%d steps)...\n', n_steps);

figure(2);
clf;
set(gcf, 'Color', 'w');

for step = 1:n_steps
    t = step * dt;
    
    % Reduced vectors (only dof)
    u_red = u_curr(freeDofs);
    v_red = v_curr(freeDofs);
    a_red = a_curr(freeDofs);
    
    % Effective load: F_eff = F + M*(a0*u + a2*v + a3*a)
    vec_pred = a0*u_red + a2*v_red + a3*a_red;
    F_eff    = F_red + M_red * vec_pred;
    
    % Solve for u_{n+1}
    u_next_red = U_eff \ (L_eff \ F_eff);
    
    % Update a_{n+1}, v_{n+1}
    a_next_red = a0*(u_next_red - u_red) - a2*v_red - a3*a_red;
    v_next_red = v_red + a6*a_red + a7*a_next_red;
    
    % Back to full vectors
    u_curr(freeDofs) = u_next_red;
    v_curr(freeDofs) = v_next_red;
    a_curr(freeDofs) = a_next_red;
    
    % Store history
    time_hist(step)     = t;
    disp_tip_hist(step) = u_curr(track_dof);
    
    % Animation every 5 steps
    if mod(step, 5) == 0
        figure(2);
        cla;
        hold on;
        
        % undeformed
        for i = 1:nel
            plot(x([conn(i,1), conn(i,2)],1), ...
                 x([conn(i,1), conn(i,2)],2), ...
                 'k:', 'linewidth', 0.5, 'Color', [0.7 0.7 0.7]);
        end
        
        % deformed
        x_def = x + reshape(u_curr, 2, [])';
        for i = 1:nel
            plot(x_def([conn(i,1), conn(i,2)],1), ...
                 x_def([conn(i,1), conn(i,2)],2), ...
                 'b-', 'linewidth', 2);
        end
        
        % truck
        plot(x_def(truck_nodes,1), x_def(truck_nodes,2), ...
             'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 6);
        
        title(sprintf('Time: %.2f s | Tip Disp: %.1f mm', ...
              t, u_curr(track_dof)*1000));
        axis equal;
        xlim([-2, 14]);
        ylim([-8, 2]);
        grid on;
        
        drawnow;
        pause(0.02);   % slow down animation
    end
end

% Print summary of transient response
[u_min, idx_min] = min(disp_tip_hist);
[u_max, idx_max] = max(disp_tip_hist);

fprintf('\nTransient response at Node 17 (vertical):\n');
fprintf('  Min disp: %8.3f mm at t = %5.3f s\n', u_min*1000, time_hist(idx_min));
fprintf('  Max disp: %8.3f mm at t = %5.3f s\n', u_max*1000, time_hist(idx_max));

% Final time history plot (no animation)
figure(3);
clf;
set(gcf, 'Color', 'w');
plot(time_hist, disp_tip_hist*1000, 'b-', 'LineWidth', 1.5);
grid on;
xlabel('Time [s]');
ylabel('Vertical displacement at Node 17 [mm]');
title('Aufgabe 2: Transient Response (Node 17, step load)');


%%  LOCAL HELPER FUNCTIONS

function [xi,w8] = gauss(nqp,ndm)
    if nqp == 1
        xi = 0; w8 = 2;
    else
        error('Wrong nqp');
    end
end 

function [N,gamma] = shape(xi,nen,ndm)
    N = zeros(nen,1); 
    gamma = zeros(nen,1); 
    if nen == 2
        N(1,1)   = 0.5*(1-xi); 
        N(2,1)   = 0.5*(1+xi);
        gamma(1) = -0.5; 
        gamma(2) =  0.5;
    else
        error('Wrong nen');
    end
end 

function [detJq,invJq] = jacobian(xe,gamma,nen,ndm)
    Jq = xe * gamma;   % scalar
    detJq = det(Jq);
    invJq = inv(Jq);
end
