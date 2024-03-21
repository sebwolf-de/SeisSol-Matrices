import sys
from seissol_matrices import json_io
from dg_matrices import dg_generator
from vtk_points import vtk_lagrange_2d, vtk_lagrange_3d

def main():
    assert len(sys.argv) > 1
    if sys.argv[1] == 'vtk':
        import dg_matrices 
        vtkpoints2 = {}
        vtkpoints3 = {}
        for deg in range(0, 8):
            vtkpoints2[deg] = vtk_lagrange_2d(deg)
            vtkpoints3[deg] = vtk_lagrange_3d(deg)
        for basisorder in range(2, 8):
            dggen = dg_generator(basisorder, 3)
            for deg in range(0, 8):
                json_io.write_matrix(dggen.collocate_volume(vtkpoints3[deg]),
                    f'coll{basisorder}vd{deg}', f'vtk{basisorder}')
                for f in range(4):
                    json_io.write_matrix(dggen.collocate_face(vtkpoints2[deg], f),
                        f'coll{basisorder}f{f}d{deg}', f'vtk{basisorder}')

if __name__ == '__main__':
    main()
