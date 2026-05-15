import pickle
import zlib
import struct
import os
import numpy as np

########################################## PART 4 : ENTROPY CODING & BINARY FILE I/O ##########################################


OUTPUT_BIN = "output.bin"


# encoded_frames → fichier .bin compressé
def write_bin(encoded_frames, output_path=OUTPUT_BIN):
    with open(output_path, 'wb') as f:
        f.write(b'MPEG4SIM')  # magic header

        # parcours des frames → sérialisation + compression zlib
        for frame_data in encoded_frames:
            ftype   = frame_data["type"]
            payload = {k: v for k, v in frame_data.items() if k != "type"}

            # conversion motion vectors → int16 pour réduire l'empreinte pickle
            if ftype == 'P' and "motion_vectors" in payload:
                payload["motion_vectors"] = np.array(
                    payload["motion_vectors"], dtype=np.int16)

            compressed = zlib.compress(pickle.dumps(payload), level=6)
            f.write(ftype.encode('ascii'))             # 1 byte : type
            f.write(struct.pack('>I', len(compressed)))  # 4 bytes : longueur
            f.write(compressed)

    file_size = os.path.getsize(output_path)
    print(f"[Part 4] Written {len(encoded_frames)} frames → '{output_path}'")
    print(f"[Part 4] Compressed size : {file_size:,} bytes  ({file_size/1024:.1f} KB)")
    return file_size


# fichier .bin → encoded_frames reconstruit
def read_bin(input_path=OUTPUT_BIN):
    encoded_frames = []

    with open(input_path, 'rb') as f:
        magic = f.read(8)
        if magic != b'MPEG4SIM':
            raise ValueError(f"Invalid file (magic={magic})")

        frame_idx = 0
        # lecture séquentielle des frames jusqu'à EOF
        while True:
            type_byte = f.read(1)
            if not type_byte:
                break  # EOF propre
            len_bytes = f.read(4)
            if len(len_bytes) < 4:
                break
            payload_len = struct.unpack('>I', len_bytes)[0]
            payload_bytes = f.read(payload_len)
            if len(payload_bytes) < payload_len:
                raise IOError(f"Truncated file at frame {frame_idx}")

            ftype   = type_byte.decode('ascii')
            payload = pickle.loads(zlib.decompress(payload_bytes))

            # restauration motion vectors numpy → liste de tuples (dy, dx)
            if ftype == 'P' and "motion_vectors" in payload:
                mv = payload["motion_vectors"]
                payload["motion_vectors"] = [
                    (int(mv[i, 0]), int(mv[i, 1])) for i in range(len(mv))]

            payload["type"] = ftype
            encoded_frames.append(payload)
            frame_idx += 1

    print(f"[Part 4] Read {len(encoded_frames)} frames from '{input_path}'")
    return encoded_frames







# taille brute Y+Cb+Cr de toutes les frames
original_size = sum(Y.nbytes + Cb.nbytes + Cr.nbytes
                    for Y, Cb, Cr in preprocessed)
print(f"[Part 4] Original raw size : {original_size:,} bytes  ({original_size/1024:.1f} KB)")

compressed_size = write_bin(encoded_frames, OUTPUT_BIN)

print(f"[Part 4] Compression ratio : {original_size / compressed_size:.2f}x")

# vérification round-trip
encoded_frames_loaded = read_bin(OUTPUT_BIN)
print(f"[Part 4] Round-trip OK : {len(encoded_frames_loaded)} frames")
print(f"  I: {sum(1 for f in encoded_frames_loaded if f['type']=='I')}  "
      f"P: {sum(1 for f in encoded_frames_loaded if f['type']=='P')}")
