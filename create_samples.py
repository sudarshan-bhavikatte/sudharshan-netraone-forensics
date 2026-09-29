import os
import cv2
from dataset import generate_synthetic_surveillance_frame, tamper_image

def create_sample_files():
    samples_dir = "./samples"
    os.makedirs(samples_dir, exist_ok=True)

    # 1. Authentic Surveillance Frame
    auth_img = generate_synthetic_surveillance_frame(640, 480, scene_type="street")
    cv2.imwrite(os.path.join(samples_dir, "sample_authentic.jpg"), auth_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    # 2. Tampered Surveillance Frame (Spliced object)
    tamp_base = generate_synthetic_surveillance_frame(640, 480, scene_type="street")
    tamp_img, mask, bbox = tamper_image(tamp_base, tamper_type="splice")
    cv2.imwrite(os.path.join(samples_dir, "sample_tampered.jpg"), tamp_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    print(f" Generated sample files in '{samples_dir}':")
    print(f" - {os.path.join(samples_dir, 'sample_authentic.jpg')}")
    print(f" - {os.path.join(samples_dir, 'sample_tampered.jpg')}")

if __name__ == "__main__":
    create_sample_files()
