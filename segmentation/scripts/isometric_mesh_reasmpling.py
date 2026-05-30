#!/usr/bin/env python3
import argparse
from pathlib import Path
import pymeshlab as ml


def process_mesh(input_path: Path, output_path: Path, iterations: int):
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist.")
        return

    print(f"Loading mesh: {input_path}...")
    ms = ml.MeshSet()
    ms.load_new_mesh(str(input_path))

    current_mesh = ms.current_mesh()
    print(
        f"Initial stats: {current_mesh.vertex_number()} vertices, {current_mesh.face_number()} faces"
    )

    # STEP 1: Isotropic Explicit Remeshing
    print(
        f"Running Isotropic Explicit Remeshing for {iterations} iterations..."
    )
    ms.meshing_isotropic_explicit_remeshing(
        iterations=iterations,
        targetlen=ml.PercentageValue(1),
        adaptive=False,
        selectedonly=False,
        featuredeg=30.0,
        checksurfdist=True,
        maxsurfdist=ml.PercentageValue(1),
        splitflag=True,
        collapseflag=True,
        swapflag=True,
        smoothflag=True,
        reprojectflag=True,
    )

    # STEP 2: Non-Manifold Repairs
    print("Repairing non-manifold topology...")

    # Remove non-manifold edges
    ms.meshing_repair_non_manifold_edges()

    # Remove non-manifold vertices with 0 displacement allowed
    # (displimit=0 guarantees vertices are split/removed without shifting positions)
    ms.meshing_repair_non_manifold_vertices(vertdispratio=0)

    # Clean up standard topological remnants left by the repairs
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_duplicate_vertices()

    # Ensure output parent directory exists before saving
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving processed mesh to: {output_path}...")
    ms.save_current_mesh(str(output_path), save_vertex_normal=False)

    final_mesh = ms.current_mesh()
    print(
        f"Success! Final stats: {final_mesh.vertex_number()} vertices, {final_mesh.face_number()} faces"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Isotropic remeshing followed by zero-displacement non-manifold repair."
    )
    parser.add_argument(
        "-i", "--input", required=True, type=Path, help="Path to input mesh"
    )
    parser.add_argument(
        "-o", "--output", required=True, type=Path, help="Path to output mesh"
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=10,
        help="Remeshing iterations",
    )

    args = parser.parse_args()

    process_mesh(
        input_path=args.input, output_path=args.output, iterations=args.iter
    )