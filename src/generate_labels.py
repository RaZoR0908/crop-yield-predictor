import os
import csv
import numpy as np
from PIL import Image

def calculate_yield(image_path, filename):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img).astype(float)

    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]

    greenness = g.mean() / (r.mean() + g.mean() + b.mean() + 1e-6)
    g_std = g.std()
    rg_ratio = r.mean() / (g.mean() + 1e-6)
    brightness = (r.mean() + g.mean() + b.mean()) / 3
    contrast = img_array.std()

    # scaled down formula so values don't cluster at top
    yield_value = (
        greenness * 6 +
        g_std / 60 +
        (1 - rg_ratio) * 1.5 +
        brightness / 300 +
        contrast / 200
    )

    # consistent noise per image using filename hash
    np.random.seed(hash(filename) % 2**32)
    noise = np.random.uniform(-1.5, 1.5)
    yield_value = yield_value + noise

    yield_value = round(min(max(yield_value, 1.0), 6.5), 2)
    return yield_value

def generate_csv(data_dir, output_csv):
    images = [f for f in os.listdir(data_dir)
              if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')]

    print(f"Found {len(images)} images")

    rows = []
    for img_name in images:
        img_path = os.path.join(data_dir, img_name)
        yield_value = calculate_yield(img_path, img_name)
        rows.append([img_name, yield_value])
        print(f"{img_name} -> yield: {yield_value}")

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'yield_tons_per_hectare'])
        writer.writerows(rows)

    print(f"\nCSV saved to {output_csv}")
    yields = [r[1] for r in rows]
    print(f"\nMin yield: {min(yields)}")
    print(f"Max yield: {max(yields)}")
    print(f"Avg yield: {round(sum(yields)/len(yields), 2)}")

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    output_csv = os.path.join(data_dir, 'labels.csv')
    generate_csv(data_dir, output_csv)