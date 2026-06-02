import pyacvd
import pyvista
from pathlib import Path
import argparse

def resample(input_mesh:Path, output_mesh:Path, cluster_resolution:int, subdivision_density:int):
    mesh = pyvista.read(str(input_mesh))

    if not mesh.is_all_triangles:
        print("Mesh needs to be triangles only")
        return
    print("Sanitizing mesh topology...")
    mesh = mesh.clean()
    
    print("Initializing pyacvd clustering...")
    _cluster = pyacvd.Clustering(mesh=mesh)

    # _cluster.subdivide(nsub=subdivision_density)
    _cluster.cluster(nclus=cluster_resolution)

    resampled_mesh = _cluster.create_mesh()
    resampled_mesh.save(str(output_mesh))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-mesh', help="Input mesh path", type=Path, required=True)
    parser.add_argument('--output-mesh', help="Output mesh path", type=Path, required=True)
    parser.add_argument('--cluster-resolution', help="Set the resolution of the final resampled mesh (Target number of vertices)", type=int, default=1_000_000)
    parser.add_argument('--subdivision-density', help="Internal subdivision for final surface mesh density", type=int, default=3)
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    input_mesh = args.input_mesh
    output_mesh = args.output_mesh
    cluster_resolution = args.cluster_resolution
    subdivision_density = args.subdivision_density

    resample(input_mesh=input_mesh, output_mesh=output_mesh, cluster_resolution=cluster_resolution, subdivision_density=subdivision_density)

if __name__ == "__main__":
    main()



