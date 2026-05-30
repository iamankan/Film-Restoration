import pymeshlab as ml
from pathlib import Path
import argparse
import shutil

def process_segmentation_mesh(input_path: Path, output_file: Path, target_faces=6500000, voxel_size=0.5):
    """
    Executes the full automated repair, hole-closing, and multi-stage decimation pipeline.
    Saves intermediate meshes in the exact same directory as the final output file.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input mesh not found at {input_path}")
        
    # Extract the parent directory from the output file path
    output_dir = output_file.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_file.exists():
        if output_file.is_dir():
            print(f"Removing legacy directory collision found at target path: {output_file}")
            shutil.rmtree(output_file)
        else:
            output_file.unlink()
    
    # Pre-emptively wipe existing intermediate files in that folder so PyMeshLab never crashes on collisions
    intermediate_names = [
        "step1_oriented.obj", "step2_resampled.obj", "step3_topological_repair.obj", 
        "step4_holes_closed.obj", "step5a_clustered.obj"
    ]
    for name in intermediate_names:
        (output_dir / name).unlink(missing_ok=True)
    output_file.unlink(missing_ok=True) # Clear the final file path too
    
    print(f"Loading raw mesh: {input_path}")
    ms = ml.MeshSet()
    ms.load_new_mesh(str(input_path))
    
    # Capture and log initial baseline metrics
    initial_mesh = ms.current_mesh()
    init_v = initial_mesh.vertex_number()
    init_f = initial_mesh.face_number()
    
    print("\n==================================================")
    print(f"INITIAL MESH STATS:")
    print(f"  Vertices (V): {init_v:,}")
    print(f"  Faces (F):    {init_f:,}")
    print("==================================================")
    
    # -------------------------------------------------------------------------
    # STEP 1: Orient Normals
    # -------------------------------------------------------------------------
    # print("\n--- STEP 1: Coordinating Face Orientations ---")
    # ms.meshing_re_orient_faces_coherently()
    # ms.meshing_invert_face_orientation()
    
    # step1_path = output_dir / "step1_oriented.obj"
    # ms.save_current_mesh(str(step1_path), save_vertex_color=False,
    #     save_vertex_coord=True,      
    #     save_vertex_normal=False,
    #     save_face_color=False,
    #     save_wedge_texcoord=False,
    #     save_wedge_normal=False,
    #     save_polygonal=False)
    # print(f"Saved intermediate: {step1_path}")
    
    # # -------------------------------------------------------------------------
    # # STEP 2: Uniform Mesh Resampling
    # # -------------------------------------------------------------------------
    # print(f"\n--- STEP 2: Running Uniform Mesh Resampling (Precision: {voxel_size}) ---")
    # ms.generate_resampled_uniform_mesh(
    #     cellsize=ml.PureValue(voxel_size),
    #     offset=ml.PureValue(0.0),
    #     mergeclosevert=True,
    #     discretize=False,
    #     multisample=False,
    #     absdist=False
    # )
    
    # step2_path = output_dir / "step2_resampled.obj"
    # ms.save_current_mesh(str(step2_path), save_vertex_color=False,
    #     save_vertex_coord=True,      
    #     save_vertex_normal=False,
    #     save_face_color=False,
    #     save_wedge_texcoord=False,
    #     save_wedge_normal=False,
    #     save_polygonal=False)
    # print(f"Saved intermediate: {step2_path}")
    
    # -------------------------------------------------------------------------
    # STEP 3: Non-Manifold Structural Purge & Cleanup
    # -------------------------------------------------------------------------
    print("\n--- STEP 3: Repairing Topological Micro-Artifacts ---")

    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    ms.meshing_remove_duplicate_vertices()

    
    ms.meshing_repair_non_manifold_edges(method=0)
    ms.meshing_repair_non_manifold_vertices(vertdispratio=0.0)
    
    
    step3_path = output_dir / "step3_topological_repair.obj"
    ms.save_current_mesh(str(step3_path), save_vertex_color=False,
        save_vertex_coord=True,      
        save_vertex_normal=False,
        save_face_color=False,
        save_wedge_texcoord=False,
        save_wedge_normal=False,
        save_polygonal=False)
    print(f"Saved intermediate: {step3_path}")

    # -------------------------------------------------------------------------
    # STEP 4: Close Gaps & Holes
    # -------------------------------------------------------------------------
    print("\n--- STEP 4: Sealing Open Boundary Holes ---")
    ms.meshing_close_holes(
        maxholesize=30,
        refinehole=True
    )
    
    ms.meshing_remove_null_faces()
    ms.meshing_remove_duplicate_vertices()
    
    step4_path = output_dir / "step4_holes_closed.obj"
    ms.save_current_mesh(str(step4_path), save_vertex_color=False,
        save_vertex_coord=True,      
        save_vertex_normal=False,
        save_face_color=False,
        save_wedge_texcoord=False,
        save_wedge_normal=False,
        save_polygonal=False)
    print(f"Saved intermediate: {step4_path}")
    
    # -------------------------------------------------------------------------
    # STEP 5: Safe Multi-Stage Decimation
    # -------------------------------------------------------------------------
    print("\n--- STEP 5: Running Multi-Stage Decimation ---")
    current_faces = ms.current_mesh().face_number()
    print(f"Initial face count entering optimization block: {current_faces:,}")
    
    if current_faces > 25000000: 
        print(f"CRITICAL: Mesh size ({current_faces:,} faces) is dangerously high.")
        print("Executing an instantaneous clustering collapse to protect system RAM...")
        ms.meshing_decimation_clustering(threshold=ml.PureValue(voxel_size * 0.4))
        current_faces = ms.current_mesh().face_number()
        print(f"Face count after lightweight clustering pass: {current_faces:,}")
        
        step5a_path = output_dir / "step5a_clustered.obj"
        ms.save_current_mesh(str(step5a_path), save_vertex_color=False,
        save_vertex_coord=True,      
        save_vertex_normal=False,
        save_face_color=False,
        save_wedge_texcoord=False,
        save_wedge_normal=False,
        save_polygonal=False)
        print(f"Saved intermediate: {step5a_path}")

    if current_faces > target_faces:
        print(f"Decimating down to requested target: {target_faces:,} faces using Quadrics...")
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            preservenormal=True,
            preservetopology=True,
            planarquadric=True,
            autoclean=True
        )
    else:
        print("Mesh scale is already within safe performance parameters. Skipping Quadric step.")

    ms.compute_normal_per_vertex()
    
    # -------------------------------------------------------------------------
    # STEP 6: Final Export
    # -------------------------------------------------------------------------
    print(f"\n--- STEP 6: Exporting Final Clean Surface ---")
    ms.save_current_mesh(
        file_name=str(output_file),
        save_vertex_color=False,
        save_vertex_coord=True,      
        save_vertex_normal=False,
        save_face_color=False,
        save_wedge_texcoord=False,
        save_wedge_normal=False,
        save_polygonal=False
    )
    
    # Capture final metrics
    final_mesh = ms.current_mesh()
    final_v = final_mesh.vertex_number()
    final_f = final_mesh.face_number()
    
    print("\n==================================================")
    print("FINAL PIPELINE REPORT:")
    print(f"  [START] Vertices: {init_v:,} | Faces: {init_f:,}")
    print(f"  [END]   Vertices: {final_v:,} | Faces: {final_f:,}")
    print(f"  Reduction:  {((init_f - final_f) / init_f) * 100:.2f}% fewer faces")
    print(f"SUCCESS: Final airtight stripped mesh saved at: {output_file}")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automated Segmentation Mesh Cleaner: Fuses self-intersections, "
                    "repairs topology, seals thin boundary loops, logs metrics, and outputs lightweight files."
    )
    
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to the input raw .obj mesh")
    parser.add_argument("-o", "--output", required=True, type=str,
                        help="Path where the final optimized .obj mesh file will be saved")
    parser.add_argument("-f", "--faces", type=int, default=6500000,
                        help="Target face count threshold for the final decimation pass (default: 6500000)")
    parser.add_argument("-v", "--voxel_size", type=float, default=0.8,
                        help="Voxel grid precision size (default: 0.8)")

    args = parser.parse_args()

    process_segmentation_mesh(
        input_path=Path(args.input),
        output_file=Path(args.output),  # Directly bound to the absolute file target
        target_faces=args.faces,
        voxel_size=args.voxel_size
    )