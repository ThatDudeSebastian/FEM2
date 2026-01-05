%--------------------------------------------------------------------------
% Problem definition
%--------------------------------------------------------------------------
ndm = 2;        % Dimension
ndf = 2;        % Dofs per element
nen = 2;        % Nodes per element

%--------------------------------------------------------------------------
% Geometry
%--------------------------------------------------------------------------


% Connectivity
conn = [
1 8;
2 8;
2 3;
3 8;
3 9;
3 4;
4 9;
4 10;
4 5;
5 10;
5 11;
5 6;
6 11;
6 12;
6 7;
7 12;
8 9;
9 10;
10 11;
11 12;  
7 13;
13 14;
13 15;
14 15;
14 16;
15 17;
];

% Number of elements
nel = size(conn, 1);


x = [0 0;
     1 0;
     2 0;
     4 0;
     6 0;
     8 0;
     9 0;
     1 0.4;
     3 0.4;
     5 0.4;
     7 0.4;
     8 0.4;
     9 -1;
     6 -2;
     12 -2;
     6 -4.5;
     12 -4.5;
    ]
    
    % Nodes
nnp = size(x, 1);

nqp = 1;                % Number of quadrature points
%--------------------------------------------------------------------------
% Materials
%--------------------------------------------------------------------------
xE = 210e9;
Area = 2*0.001;
eta = 1e11;


%--------------------------------------------------------------------------
% Boundary conditions
%--------------------------------------------------------------------------

% Dirichlet boundary condition
%      node  ldof  loadid     scale
drlt = [
1, 1, 2, 0;
1, 2, 2, 0;
2, 1, 2, 0;
2, 2, 2, 0;
];

% Neumann boundary condition
%      node  ldof  loadid  scale
neum = [
  16, 2, 2, -3090*9.81;
  17, 2, 2, -3090*9.81;
];

% loadcurves
loadcurve(1).time =         [0        3];
loadcurve(1).value =         [0        0];
loadcurve(2).time = [0
                        1
                        2
                        3
                        4
                        5
]';
loadcurve(2).value = [0
                        1
                        1
                        1
                        1
                        1
]';

% time step / load step variables
ttime = 5; % total time of simulation
dt    = 0.1; % time step

% BC variables 
allDofs = (1:1:nnp*ndf)';       % array with all DOF numbers
numDrltDofs = size(drlt,1);     % number of dirichlet DOF's
drltDofs = zeros(numDrltDofs,1);% initialise drltDofs array

clf;
xplt = x;
for i = 1:size(conn, 1)
  plot (xplt([conn(i,1), conn(i,2)],1), xplt([conn(i,1), conn(i,2)],2), 'linewidth', 3, 'x-')
  axis equal
  hold on
endfor