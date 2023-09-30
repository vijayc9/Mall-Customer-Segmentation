import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


class EmbeddingGenerator:
    def __init__(
        self, folder_path, batch_size=32, model_type="dinov2_vitl14", device=None
    ):
        self.folder_path = folder_path
        self.batch_size = batch_size

        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(self.device)

        self.transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.model = torch.hub.load("facebookresearch/dinov2", model_type).to(
            self.device
        )

        data = self._get_embeddings_for_folder()
        json_filename = f"{self.folder_path}/{self.folder_path.split('/')[-1]}.json"
        with open(json_filename, "w") as json_file:
            json.dump(data, json_file, indent=4)

    def preprocess(self, image_path):
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transforms(image).unsqueeze(0)

    def _run_model(self, image_tensor):
        with torch.no_grad():
            features = self.model(image_tensor.to(self.device))
            features = torch.nn.functional.normalize(features, dim=1)

        return features.detach().cpu().numpy()

    def _get_embeddings_for_folder(self):
        folders = [
            item
            for item in os.listdir(self.folder_path)
            if os.path.isdir(os.path.join(self.folder_path, item))
        ]

        data = []
        for folder in folders:
            filenames = [
                f
                for f in os.listdir(os.path.join(self.folder_path, folder))
                if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".tif")
            ]

            num_files = len(filenames)
            for i in tqdm(
                range(0, num_files, self.batch_size), desc=f"{folder}({num_files})"
            ):
                batch_files = filenames[i : i + self.batch_size]
                actual_batch_size = len(batch_files)

                batch_tensors = [
                    self.preprocess(os.path.join(self.folder_path, folder, f))
                    for f in batch_files
                ]
                batch_tensor = torch.cat(batch_tensors, 0)

                embeddings = self._run_model(batch_tensor)

                for j in range(actual_batch_size):
                    t = {
                        "id": batch_files[j].split(".")[0],
                        "filepath": os.path.join(
                            self.folder_path, folder, batch_files[j]
                        ),
                        "label": folder,
                        "embedding": embeddings[j].tolist(),
                    }

                    data.append(t)
        return data


if __name__ == "__main__":
    folder_path = "/Users/vijaychaurasiya/Desktop/Satsure/dataset/small"
    embedding_generator = EmbeddingGenerator(folder_path, batch_size=32)
