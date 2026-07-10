// Thin 3D extrusion for an OpenFOAM-style 2D plane-channel case.
// All dimensions are SI metres. front/back become empty patches after import.
L = 1.0;
H = 0.02;
W = 0.001;
Nx = 100;
Ny = 20;

Point(1) = {0, -H/2, -W/2};
Point(2) = {L, -H/2, -W/2};
Point(3) = {L,  H/2, -W/2};
Point(4) = {0,  H/2, -W/2};
Point(5) = {0, -H/2,  W/2};
Point(6) = {L, -H/2,  W/2};
Point(7) = {L,  H/2,  W/2};
Point(8) = {0,  H/2,  W/2};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};
Line(9) = {1, 5};
Line(10) = {2, 6};
Line(11) = {3, 7};
Line(12) = {4, 8};

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Curve Loop(2) = {1, 10, -5, -9};
Plane Surface(2) = {2};
Curve Loop(3) = {2, 11, -6, -10};
Plane Surface(3) = {3};
Curve Loop(4) = {3, 12, -7, -11};
Plane Surface(4) = {4};
Curve Loop(5) = {4, 9, -8, -12};
Plane Surface(5) = {5};
Curve Loop(6) = {5, 6, 7, 8};
Plane Surface(6) = {6};

Surface Loop(1) = {1, 2, 3, 4, 5, 6};
Volume(1) = {1};

Transfinite Curve {1, 3, 5, 7} = Nx + 1;
Transfinite Curve {2, 4, 6, 8} = Ny + 1;
Transfinite Curve {9, 10, 11, 12} = 2;
Transfinite Surface {1, 2, 3, 4, 5, 6};
Recombine Surface {1, 2, 3, 4, 5, 6};
Transfinite Volume {1};

Physical Surface("front") = {1};
Physical Surface("wall") = {2, 4};
Physical Surface("outlet") = {3};
Physical Surface("inlet") = {5};
Physical Surface("back") = {6};
Physical Volume("fluid") = {1};

Mesh.MshFileVersion = 2.2;
Mesh.Binary = 0;
Mesh.ElementOrder = 1;
