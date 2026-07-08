SetFactory("OpenCASCADE");

// Simple SI pipe mesh for FerrumCFD import and OpenFOAM comparison.
// Units: m, s, kg, K. The boundary layer field creates two near-wall layers.
// Parameters can be overridden from the CLI, for example:
// gmsh -3 pipe_prism2.geo -format msh2 -setnumber axial_cells 48 -o pipe.msh

DefineConstant[
    radius = {0.01, Min 0.0001, Max 10.0, Name "FerrumCFD/pipe/radius"},
    length = {1.0, Min 0.0001, Max 1000.0, Name "FerrumCFD/pipe/length"},
    lc_center = {0.004, Min 0.00001, Max 1.0, Name "FerrumCFD/mesh/lc_center"},
    lc_wall = {0.0012, Min 0.00001, Max 1.0, Name "FerrumCFD/mesh/lc_wall"},
    axial_cells = {32, Min 1, Max 4096, Step 1, Name "FerrumCFD/mesh/axial_cells"},
    bl_hwall = {0.00045, Min 0.000001, Max 1.0, Name "FerrumCFD/boundaryLayer/hwall_n"},
    bl_hfar = {0.003, Min 0.000001, Max 1.0, Name "FerrumCFD/boundaryLayer/hfar"},
    bl_thickness = {0.0015, Min 0.000001, Max 1.0, Name "FerrumCFD/boundaryLayer/thickness"},
    bl_ratio = {1.25, Min 1.0, Max 10.0, Name "FerrumCFD/boundaryLayer/ratio"},
    bl_layers = {2, Min 1, Max 20, Step 1, Name "FerrumCFD/boundaryLayer/layers"}
];

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
Field[1].hwall_n = bl_hwall;
Field[1].hfar = bl_hfar;
Field[1].thickness = bl_thickness;
Field[1].ratio = bl_ratio;
Field[1].Quads = 1;
Field[1].NbLayers = bl_layers;
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
