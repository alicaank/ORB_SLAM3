import sys
from pathlib import Path

def generate_depth_txt(folder: Path, output_filename="depth.txt"):
    # Get all png images from the folder and sort them
    image_files = sorted(folder.glob("*.png"))
    if not image_files:
        print("No PNG images found in folder:", folder)
        return

    # Prepare header lines.
    header = [
        "# depth maps",
        f"# file: '{folder.name}.bag'",
        "# timestamp filename"
    ]
    
    lines = header.copy()
    for img in image_files:
        # Assume filename is like '<timestamp>.png'
        timestamp = img.stem
        # Construct a relative path line with "depth/" prefix
        rel_path = f"depth/{img.name}"
        lines.append(f"{timestamp} {rel_path}")

    # Write the output file in the given folder.
    output_path = folder / output_filename
    with output_path.open("w") as f:
        f.write("\n".join(lines))
    print(f"Generated {output_path}")

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} path_to_folder")
        sys.exit(1)
    folder = Path(sys.argv[1])
    if not folder.is_dir():
        print("Provided path is not a folder!")
        sys.exit(1)
    generate_depth_txt(folder)

if __name__ == "__main__":
    main()