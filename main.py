from PIL import Image
import struct
import os

for frame_number in range(100):
    input_bin = f"output_{frame_number}.data"
    output_png = f"/res/{frame_number}.png"

    try:
        # Check if file exists
        if not os.path.exists(input_bin):
            print(f"File {input_bin} not found!")
            continue
            
        with open(input_bin, "rb") as fin:
            # Read file content
            data = fin.read()
            
            if len(data) < 8:
                print(f"Error: File {input_bin} is too small to contain width and height")
                continue
                
            # Extract width and height
            w, h = struct.unpack("<ii", data[:8])
            print(f"Width: {w}, Height: {h}")
            
            # Calculate expected pixel data size
            expected_size = 4 * w * h
            
            if len(data) - 8 < expected_size:
                print(f"Warning: File has {len(data) - 8} bytes of pixel data, but {expected_size} bytes are expected")
                # Adjust width or height to match available data
                available_pixels = (len(data) - 8) // 4  # Number of complete RGBA pixels
                if available_pixels < w * h:
                    # Recalculate dimensions to fit available data
                    if w > h:
                        w = available_pixels // h
                    else:
                        h = available_pixels // w
                    print(f"Adjusted dimensions to Width: {w}, Height: {h}")
            
            # Extract pixel data
            pixels = data[8:8 + 4 * w * h]

        # Create image
        img = Image.frombytes("RGBA", (w, h), pixels)
        img.save(output_png)
        print(f"Frame {frame_number} successfully saved to {output_png}!")

    except Exception as e:
        print(f"Error processing {input_bin}: {str(e)}")