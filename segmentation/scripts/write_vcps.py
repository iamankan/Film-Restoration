import numpy as np

def write_vcps(filename, points):
    
    with open(filename, 'wb') as f:
        # Write ASCII header
        f.write(f"width: {points.shape[0]}\n".encode('ascii'))
        f.write(f"height: {1}\n".encode('ascii'))
        f.write(f"dim: {points.shape[1]}\n".encode('ascii'))
        f.write(f"ordered: true\n".encode('ascii'))
        f.write(f"type: double\n".encode('ascii'))
        f.write(f"version: 1\n".encode('ascii'))
        f.write(b"<>\n")
        
        f.write(points.astype(np.double).tobytes())

# Example usage:
points = np.array([
    [1.1, 2.2, 3.3],
    [4.4, 5.5, 6.6],
    [1.1, 2.2, 3.3],
    [4.4, 5.5, 6.6],
    [1.1, 2.2, 3.3],
    [4.4, 5.5, 6.6],
    [1.1, 2.2, 3.3],
    [4.4, 5.5, 6.6]
], dtype=np.double)
print(points.shape, points.size)
write_vcps('/home/ankan/Downloads/example.vcps', points)