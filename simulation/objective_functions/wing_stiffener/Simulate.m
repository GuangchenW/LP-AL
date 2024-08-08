function displacement = Simulate(E,d,P1,P2,F1,F2,F3,F4,F5,F6)
%SIMULATE Run static plane stress simulation with the given parameters.
%   All input parameters are signless (magnitude).
%   Input parameters:
%   E  - Young's modulus (GPa)
%   d  - Thickness of the material (mm)
%   P1 - Pressure applied on lower edge (inward)
%   P2 - Pressure applied on upper edge (outward)
%   F1 - X-Force H4 (mean 23758)
%   F2 - Y-Force H4 (mean 35239)
%   F3 - X-Force H7 (mean -5949)
%   F4 - Y-Force H7 (mean -16245)
%   F5 - X-Force H8 (mean 19185)
%   F6 - Y-Force H8 (mean -10140)

E = E*1e9;
nu = 0.3; % Poisson's ratio (constant)
d = d*1e-3; % Convert mm to m

model = LoadModel(); % Load the model
model.BoundaryConditions = []; % Clear all BCs
structuralProperties(model,'YoungsModulus',E,'PoissonsRatio',nu); % Assign properties

% Fixature
structuralBC(model,'Edge',1,'Constraint','fixed');

% Areodynamic load
structuralBoundaryLoad(model,"Edge",2,"Pressure",P1);
structuralBoundaryLoad(model,"Edge",3,"Pressure",-P2); % Outward so negative

% Hole area
H4Area = d*0.085*0.085*pi;
H7Area = d*0.3325*0.3325*pi;
H8Area = H4Area;

% Pressure on holes
H4P = [F1;F2]/H4Area;
H7P = [-F3;-F4]/H7Area;
H8P = [F5;-F6]/H8Area;

% Apply pressure to holes
structuralBoundaryLoad(model,'Edge',4,'SurfaceTraction',H4P);
structuralBoundaryLoad(model,'Edge',7,'SurfaceTraction',H7P);
structuralBoundaryLoad(model,'Edge',8,'SurfaceTraction',H8P);

% Create mesh
generateMesh(model, Hmax=0.1);

% Solve for displacement
result = solve(model);
displacement = max(result.Displacement.uy);
end

