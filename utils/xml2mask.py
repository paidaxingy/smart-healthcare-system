import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm

def xmls_to_masks(img_dir, xml_dir, mask_dir):
    os.makedirs(mask_dir, exist_ok=True)
    for img_name in tqdm(os.listdir(img_dir)):
        if not img_name.endswith('.tif'):
            continue
        base = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        xml_path = os.path.join(xml_dir, base + '.xml')
        mask_path = os.path.join(mask_dir, base + '.png')

        if not os.path.exists(xml_path):
            print(f"Warning: {xml_path} 不存在，跳过。")
            continue

        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        tree = ET.parse(xml_path)
        root = tree.getroot()
        for region in root.iter('Region'):
            coords = []
            for vertex in region.iter('Vertex'):
                x = float(vertex.attrib['X'])
                y = float(vertex.attrib['Y'])
                coords.append([int(round(x)), int(round(y))])
            if len(coords) > 2:
                coords = np.array([coords], dtype=np.int32)
                cv2.fillPoly(mask, coords, 255)
        cv2.imwrite(mask_path, mask)
    print(f'{mask_dir} 全部mask已生成！')

if __name__ == "__main__":
    # 处理Training
    xmls_to_masks(
        img_dir='data/Training/Tissue Images',
        xml_dir='data/Training/Annotations',
        mask_dir='data/Training/Masks'
    )
    # 处理Test
    xmls_to_masks(
        img_dir='data/Test',
        xml_dir='data/Test',
        mask_dir='data/Test/Masks'
    )