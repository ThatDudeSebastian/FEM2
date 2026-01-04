%% Aufgabe 3 - Quasi-static elastoplastic 2D truss (crane)
% Residual form + Newton-Raphson + load history interpolation
% Includes a simple design loop for cross-sectional area (just yielding, not exceeding sigma_u)

clear; clc; close all;
format short e;

%% =========================
%  1) INPUT: GEOMETRY / MODEL
% =========================
ndm = 2;      % spatial dimension
ndf = 2;      % dof per node (ux, uy)

% Connectivity (Crane + Hanging/Truck part)
conn = [
    1 8;  2 8;  2 3;  3 8;  3 9;  3 4;  4 9;  4 10; 4 5;  5 10;
    5 11; 5 6;  6 11; 6 12; 6 7;  7 12; 8 9;  9 10; 10 11; 11 12;
    7 13; 13 14; 13 15; 14 15; 14 16; 15 17
];

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

nnp  = size(x,1);
ndof = nnp*ndf;

truck_nodes = [16 17];

% -------------------------------------------------------
% Stabilization bracing for the lower frame (pure truss frames can be mechanisms)
% Default: stable configuration (recommended).
% -------------------------------------------------------
useStabilization = true;
if useStabilization
    conn_add = [16 17; 14 17; 15 16];   % bottom chord + two diagonals
    conn = [conn; conn_add];
    fprintf('INFO: Added stabilization bracing elements.\n');
end
nel = size(conn,1);

%% =========================
%  2) MATERIAL (Elasto-plastic 1D bar)
% =========================
mat.E        = 210e9;        % Young's modulus [Pa]
mat.sigma_y0 = 960e6;        % initial yield stress [Pa]
mat.H        = 2.0e9;        % isotropic hardening modulus [Pa]
mat.sigma_u  = 1.10e9;       % tensile strength [Pa] (must not be exceeded)

% Cross-section (initial)
r  = 0.035;                 % [m]
A0 = 2*pi*r^2;              % [m^2] (example choice)
mat.A = A0;

%% =========================
%  3) LOAD: truck weight (quasi-static scaling)
% =========================
g = 9.81;                   % [m/s^2]
truck_mass_total = 4100;    % [kg]
mass_per_node    = truck_mass_total / numel(truck_nodes);

Fref = zeros(ndof,1);
for k = 1:numel(truck_nodes)
    node = truck_nodes(k);
    dofY = (node-1)*ndf + 2;
    Fref(dofY) = Fref(dofY) - mass_per_node*g;
end

% Load history: times and scaling factors (linearly interpolated)
t_path      = [0.0  1.0];
lambda_path = [0.0  1.2];

nSteps  = 30;
tSteps  = linspace(t_path(1), t_path(end), nSteps);
lambdaSteps = interp1(t_path, lambda_path, tSteps, 'linear');

%% =========================
%  4) BOUNDARY CONDITIONS
% =========================
% Essential BC: Node 1 and 2 fixed (ux, uy)
drlt = [
    1 1; 1 2;
    2 1; 2 2
];

% Extra constraints to remove rigid body / pendulum-like mechanisms
drlt = [drlt; 13 1];   % fix node 13 in x (prevents lateral sway of the hanger joint)
drlt = [drlt; 16 1];   % fix node 16 in x (prevents free rotation/swing of the hanging frame)

drltDofs = unique( (drlt(:,1)-1)*ndf + drlt(:,2) );
allDofs  = (1:ndof)';
freeDofs = setdiff(allDofs, drltDofs);

fprintf('\nAufgabe 3: Quasi-static elastoplastic 2D truss (Residual form + NR)\n');
fprintf('Elements: %d, Nodes: %d, Free DOFs: %d\n\n', nel, nnp, numel(freeDofs));

%% =========================
%  5) PRE-CHECK (elastic stiffness conditioning)
% =========================
u0 = zeros(ndof,1);
state0 = initState(nel);
[K0, ~, ~] = assembleGlobal(u0, state0, x, conn, mat, true);
Kff0 = K0(freeDofs, freeDofs);
rc0  = rcond(Kff0);
fprintf('Initial rcond(Kff0) = %.3e\n', rc0);
if rc0 < 1e-12
    warning('Initial stiffness is (near) singular. Consider adding bracing or constraints.');
end
fprintf('\n');

%% =========================
%  6) DESIGN: choose Area so it "just yields" but does not exceed sigma_u
% =========================
doDesignToYield = true;

targetSigma  = mat.sigma_y0 * 1.05;  % slightly above yield to activate plasticity
maxDesignIts = 15;
tolDesignRel = 0.01;                % 1% tolerance

if doDesignToYield
    fprintf('--- Design loop: scale area so max stress ~ target (and <= sigma_u) ---\n');
    A_scale = 1.0;

    for itD = 1:maxDesignIts
        mat_it = mat;
        mat_it.A = A0 * A_scale;

        result = solveQuasiStaticEP(x, conn, mat_it, Fref, freeDofs, lambdaSteps);

        sigmaMax = result.sigmaMax_final;
        if isnan(sigmaMax) || isinf(sigmaMax)
            error('Simulation failed (NaN/Inf). Likely mechanism or divergence.');
        end

        relErr = (sigmaMax - targetSigma)/targetSigma;
        fprintf('Design It %2d: A_scale=%.4f | MaxSigma=%.3e | Target=%.3e | Err=%.2f%%\n', ...
            itD, A_scale, sigmaMax, targetSigma, 100*relErr);

        % Enforce tensile strength constraint
        if sigmaMax > mat.sigma_u
            A_scale = A_scale * (sigmaMax/mat.sigma_u);
            fprintf('  -> Above sigma_u. Increasing area.\n');
            continue;
        end

        if abs(relErr) < tolDesignRel
            fprintf('>>> Design converged.\n\n');
            mat = mat_it;
            break;
        end

        % Stress ~ 1/A heuristic update with mild damping
        A_scale_new = A_scale * (sigmaMax/targetSigma);
        A_scale = 0.6*A_scale + 0.4*A_scale_new;

        if itD == maxDesignIts
            fprintf('>>> Max design iterations reached. Using last A_scale.\n\n');
            mat = mat_it;
        end
    end
else
    result = solveQuasiStaticEP(x, conn, mat, Fref, freeDofs, lambdaSteps);
end

%% =========================
%  7) FINAL RUN (history for plotting)
% =========================
result = solveQuasiStaticEP(x, conn, mat, Fref, freeDofs, lambdaSteps);

% Plot 1: Load factor vs vertical displacement at node 17
trackNode = 17;
trackDofY = (trackNode-1)*ndf + 2;

figure(1); clf; set(gcf,'Color','w');
plot(lambdaSteps, 1000*result.uHist(trackDofY,:), '.-','LineWidth',1.5);
grid on;
xlabel('Load factor \lambda');
ylabel('u_y(Node 17) [mm]');
title('Load-Displacement (Quasi-static NR)');

% Plot 2: Undeformed vs deformed shape (yielded elements in red)
figure(2); clf; set(gcf,'Color','w'); hold on;

plotTruss(x, conn, 'k:', 0.8);  % undeformed

uFinal   = result.uHist(:, end);
scaleDef = 5.0;                % visual scaling only
xDef = x + reshape(uFinal,2,[])' * scaleDef;

yieldedIdx = find(result.yieldedFinal);
elasticIdx = find(~result.yieldedFinal);

plotTrussSubset(xDef, conn, elasticIdx, 'b-', 1.5);
if ~isempty(yieldedIdx)
    plotTrussSubset(xDef, conn, yieldedIdx, 'r-', 3.0);
    legend('Original','Elastic','Yielded','Location','best');
else
    legend('Original','Elastic','Location','best');
end

plot(xDef(truck_nodes,1), xDef(truck_nodes,2), 'ro', 'MarkerFaceColor','r');
axis equal; grid on;
title(sprintf('Final Deformation (x%.1f) | Red = Yielded', scaleDef));

%% Final report
fprintf('=== FINAL RESULTS ===\n');
fprintf('Area A     = %.6e m^2 (scale %.4f)\n', mat.A, mat.A/A0);
fprintf('Max sigma  = %.3e Pa\n', result.sigmaMax_final);
fprintf('sigma_y0   = %.3e Pa\n', mat.sigma_y0);
fprintf('sigma_u    = %.3e Pa\n', mat.sigma_u);

if result.sigmaMax_final >= mat.sigma_y0
    fprintf('Status: YIELDING OCCURRED (plasticity active).\n');
else
    fprintf('Status: ELASTIC (no yielding).\n');
end
if result.sigmaMax_final > mat.sigma_u
    fprintf('WARNING: Tensile strength exceeded!\n');
end

%% =========================
%  LOCAL FUNCTIONS
% =========================

function result = solveQuasiStaticEP(x, conn, mat, Fref, freeDofs, lambdaSteps)
    nnp  = size(x,1);
    ndof = nnp*2;
    nel  = size(conn,1);

    state_n = initState(nel);      % committed state
    u_n     = zeros(ndof,1);       % committed displacements

    uHist = zeros(ndof, numel(lambdaSteps));
    sigmaMaxHist = zeros(1, numel(lambdaSteps));

    maxNR = 25;
    tolR  = 1e-8;

    for s = 1:numel(lambdaSteps)
        lam = lambdaSteps(s);

        u = u_n;  % predictor

        converged = false;
        for it = 1:maxNR
            [K, fint, state_trial] = assembleGlobal(u, state_n, x, conn, mat, false);

            % Residual: R(u) = fint(u) - lam*Fref = 0
            R  = fint - lam*Fref;
            Rf = R(freeDofs);

            normRef = max(1, norm(lam*Fref(freeDofs)));
            if norm(Rf) < tolR * normRef
                u_n     = u;
                state_n = state_trial;
                converged = true;
                break;
            end

            Kff = K(freeDofs, freeDofs);

            % Optional conditioning warning (MATLAB may warn anyway)
            if rcond(Kff) < 1e-14
                warning('Matrix is close to singular or badly scaled. Results may be inaccurate. RCOND = %.3e', rcond(Kff));
            end

            du = - Kff \ Rf;
            u(freeDofs) = u(freeDofs) + du;
        end

        if ~converged
            warning('NR did not converge at step %d (lambda=%.3f). Committing last iterate.', s, lam);
            [~, ~, state_trial] = assembleGlobal(u, state_n, x, conn, mat, false);
            u_n = u;
            state_n = state_trial;
        end

        uHist(:,s) = u_n;
        sigmaMaxHist(s) = max(abs(state_n.sigma));
    end

    result.uHist = uHist;
    result.sigmaMax_final = sigmaMaxHist(end);
    result.sigmaMaxHist = sigmaMaxHist;
    result.yieldedFinal = state_n.yielded;
end

function [K, fint, state_out] = assembleGlobal(u, state_n, x, conn, mat, elasticOnly)
    if nargin < 6, elasticOnly = false; end

    ndof = size(x,1)*2;
    K    = zeros(ndof, ndof);
    fint = zeros(ndof, 1);

    state_out = state_n;

    for e = 1:size(conn,1)
        nodes = conn(e,:);
        idx = [nodes(1)*2-1, nodes(1)*2, nodes(2)*2-1, nodes(2)*2];

        xe = x(nodes,:)';
        dx = xe(1,2) - xe(1,1);
        dy = xe(2,2) - xe(2,1);

        L = sqrt(dx^2 + dy^2);
        c = dx / L;
        s = dy / L;

        ue  = u(idx);
        B   = [-c -s c s] / L;
        eps = B * ue;

        if elasticOnly
            sig = mat.E * (eps - state_n.ep(e));
            Et  = mat.E;
            ep_new = state_n.ep(e);
            a_new  = state_n.alpha(e);
            yielded = false;
        else
            [sig, Et, ep_new, a_new, yielded] = updateMat(eps, state_n.ep(e), state_n.alpha(e), mat);
        end

        state_out.ep(e)      = ep_new;
        state_out.alpha(e)   = a_new;
        state_out.sigma(e)   = sig;
        state_out.yielded(e) = yielded;

        fint(idx)   = fint(idx)   + (B' * sig) * mat.A * L;
        K(idx, idx) = K(idx, idx) + (B' * Et  * B) * mat.A * L;
    end
end

function [sig, Et, ep, alpha, yielded] = updateMat(eps, ep_old, alpha_old, mat)
    % 1D elastoplasticity with isotropic hardening (return mapping)

    sig_tr = mat.E * (eps - ep_old);
    f = abs(sig_tr) - (mat.sigma_y0 + mat.H * alpha_old);

    if f <= 0
        sig = sig_tr;
        Et  = mat.E;
        ep  = ep_old;
        alpha = alpha_old;
        yielded = false;
    else
        dgam = f / (mat.E + mat.H);

        sig = sig_tr - sign(sig_tr) * mat.E * dgam;
        ep  = ep_old + dgam * sign(sig_tr);
        alpha = alpha_old + dgam;

        Et = (mat.E * mat.H) / (mat.E + mat.H);
        yielded = true;
    end
end

function s = initState(nel)
    s.ep      = zeros(nel,1);
    s.alpha   = zeros(nel,1);
    s.sigma   = zeros(nel,1);
    s.yielded = false(nel,1);
end

function plotTruss(x, conn, style, lw)
    for e = 1:size(conn,1)
        pts = x(conn(e,:), :);
        plot(pts(:,1), pts(:,2), style, 'LineWidth', lw);
    end
end

function plotTrussSubset(x, conn, idxList, style, lw)
    for k = 1:numel(idxList)
        e = idxList(k);
        pts = x(conn(e,:), :);
        plot(pts(:,1), pts(:,2), style, 'LineWidth', lw);
    end
end
