from locust import HttpUser, task, between
import os

# Create a small test image once
import struct, zlib

def make_test_png():
    def png_chunk(name, data):
        c = zlib.crc32(name + data) & 0xffffffff
        return struct.pack('>I', len(data)) + name + data + struct.pack('>I', c)
    raw = b'\x89PNG\r\n\x1a\n'
    raw += png_chunk(b'IHDR', struct.pack('>IIBBBBB', 224, 224, 8, 2, 0, 0, 0))
    raw_data = b'\x00' + b'\xff\x88\x44' * 224
    raw_data = raw_data * 224
    compressed = zlib.compress(raw_data)
    raw += png_chunk(b'IDAT', compressed)
    raw += png_chunk(b'IEND', b'')
    return raw

TEST_IMAGE = make_test_png()

class PredictUser(HttpUser):
    wait_time = between(0.5, 2)

    @task(3)
    def predict(self):
        self.client.post("/predict",
            files={"file": ("retina.png", TEST_IMAGE, "image/png")})

    @task(1)
    def health(self):
        self.client.get("/health")

    @task(1)
    def metrics(self):
        self.client.get("/metrics")
