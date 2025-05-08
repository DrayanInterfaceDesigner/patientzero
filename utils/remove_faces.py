import os

def process_off_file(input_path, output_path):
    with open(input_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if not lines or lines[0].upper() != 'OFF':
        return False

    # Parse header (handle both single-line and two-line headers)
    if lines[0].upper() == 'OFF' and len(lines) > 1:
        counts = lines[1].split()
    else:
        counts = lines[0].split()[1:]

    if len(counts) < 2:
        return False

    try:
        n_vertices = int(counts[0])
        n_faces = int(counts[1])
    except ValueError:
        return False

    # Find vertex data (skip header lines)
    vertex_start = 2 if (lines[0].upper() == 'OFF' and len(lines) > 1) else 1
    vertices_end = vertex_start + n_vertices

    if len(lines) < vertices_end:
        return False  # Not enough vertex lines

    # Build new content
    new_content = [f"OFF\n{n_vertices} 0 0\n"]
    new_content.extend([f"{line}\n" for line in lines[vertex_start:vertices_end]])

    with open(output_path, 'w') as f:
        f.writelines(new_content)
    
    return True

def main():
    input_folder = input("Enter the path to the folder containing OFF files: ").strip()
    output_folder = input("Enter the output path for point clouds: ").strip()
    
    os.makedirs(output_folder, exist_ok=True)
    
    processed = 0
    errors = 0
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.off'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            print(f"Processing {filename}...", end=' ')
            if process_off_file(input_path, output_path):
                print("Done")
                processed += 1
            else:
                print("Failed")
                errors += 1
    
    print(f"\nResults: {processed} successful, {errors} failed")

if __name__ == "__main__":
    print("OFF to Point Cloud Converter")
    main()