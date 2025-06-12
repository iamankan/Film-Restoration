import struct

def read_vcps(filepath):
    with open(filepath, 'rb') as f:
        
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Unexpected end of file before binary section.")
            if line.strip() == b'<>':
                break
            header_lines.append(line.decode('ascii').strip())


        header = dict()
        for line in header_lines:
            if ':' in line:
                key, value = line.split(':', 1)
                header[key.strip()] = value.strip()

        width = int(header['width'])
        dim = int(header['dim'])
        dtype = header['type']

        if dtype != 'double':
            raise NotImplementedError(f"Unsupported data type: {dtype}")

        num_values = width * dim
        byte_data = f.read(num_values * 8)  # double = 8 bytes
        values = struct.unpack('<' + 'd' * num_values, byte_data)

        # Step 4: Convert flat list to (x, y, z) tuples
        points = [tuple(values[i:i+dim]) for i in range(0, len(values), dim)]
        return points, values


# points, values = read_vcps("/media/ankan/Ankan_PhD/IlFord/EduceMount/Xometry_MJF_VP/FrameAvg/volpkgs/001_IBW_10um_60kV_MS.volpkg/paths/20250424171713/pointset.vcps")
points, values = read_vcps("/home/ankan/Downloads/example.vcps")

for p in points[:5]:
    print(p)

