import pymeshlab
import argparse
import shutil
from pathlib import Path

def split_mesh_into_components(input_file, output_dir):
    input_path = Path(input_file)
    out_path = Path(output_dir)

    # 1. Handle output directory: create if not exists, empty if it does
    if out_path.exists():
        print(f"Cleaning existing directory: {out_path}")
        for item in out_path.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {out_path}")

    # 2. Load the source mesh
    # We keep this ms object pristine
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(input_path))
    
    # 3. Use the correct filter to split the mesh
    # This generates a new MeshSet containing all components as separate meshes
    print("Splitting mesh into connected components...")
    split_ms = ms.generate_splitting_by_connected_components()
    
    num_components = split_ms.mesh_number()
    print(f"Successfully split into {num_components} components.")

    # 4. Save each component
    for i in range(num_components):
        # We access the i-th mesh in the new set
        component_mesh = split_ms.get_mesh(i)
        
        # Create a temporary container for saving
        temp_ms = pymeshlab.MeshSet()
        temp_ms.add_mesh(component_mesh)
        
        output_file = out_path / f"mesh_cc_{i}.obj"
        temp_ms.save_current_mesh(str(output_file))
        print(f"Saved: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split OBJ into connected components using Pymeshlab.")
    parser.add_argument("-i", "--input", required=True, help="Path to input OBJ file")
    parser.add_argument("-o", "--output-dir", required=True, help="Directory to save output files")
    
    args = parser.parse_args()
    
    split_mesh_into_components(args.input, args.output_dir)