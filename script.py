import modal
from pathlib import Path


from sam3_table.training_config import SAM3LoRAConfig
from sam3_table.cstone_train_sam3 import train_sam3, app
from sam3_table.coco_schema import COCODataset
from PIL import Image

def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path)

def load_coco_dataset(coco_path: str) -> COCODataset:
    return COCODataset.from_json(coco_path)

if __name__ == "__main__":
    path = Path(__file__).resolve().parent / "sam3_table" / "testSamples" / "full_lora_config.yaml"
    config = SAM3LoRAConfig.from_yaml(path)

    coco_path = Path(__file__).resolve().parent / "sam3_table" / "testSamples" / "ex_annotations.coco.json"
    coco_dataset = load_coco_dataset(str(coco_path))

    with app.run():
        print(train_sam3.remote(config, coco_dataset))