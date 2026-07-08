SetFactory("OpenCASCADE");

// Simple SI pipe mesh for FerrumCFD import and OpenFOAM comparison.
// Units: m, s, kg, K. The boundary layer field creates two near-wall layers.

radius = 0.01;
length = 1.0;

lc_center = 0.004;
lc_wall = 0.0012;
axial_cells = 32;

Point(1) = {0, radius, 0, lc_wall};
Point(2) = {0, 0, radius, lc_wall};
Point(3) = {0, -radius, 0, lc_wall};
Point(4) = {0, 0, -radius, lc_wall};
Point(5) = {0, 0, 0, lc_center};

Circle(1) = {1, 5, 2};
Circle(2) = {2, 5, 3};
Circle(3) = {3, 5, 4};
Circle(4) = {4, 5, 1};

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Point {5} In Surface {1};

Field[1] = BoundaryLayer;
Field[1].EdgesList = {1, 2, 3, 4};
Field[1].hwall_n = 0.00045;
Field[1].hfar = 0.003;
Field[1].thickness = 0.0015;
Field[1].ratio = 1.25;
Field[1].Quads = 1;
Field[1].NbLayers = 2;
BoundaryLayer Field = 1;

Physical Surface("inlet", 1) = {1};

extr[] = Extrude {length, 0, 0} {
    Surface{1};
    Layers {axial_cells};
    Recombine;
};

Physical Surface("outlet", 2) = {extr[0]};
Physical Surface("wall", 3) = {extr[2], extr[3], extr[4], extr[5]};
Physical Volume("fluid", 10) = {extr[1]};

Mesh.Algorithm = 6;
Mesh.Algorithm3D = 10;
Mesh.RecombinationAlgorithm = 1;
Mesh.Smoothing = 10;
Mesh.ElementOrder = 1;
