import tempfile

import diskcache
import torch
from PIL import Image
from signwriting.visualizer.visualize import signwriting_to_image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from signwriting_evaluation.metrics.base import SignWritingMetric


def signwriting_to_clip_image(fsw: str, size=224) -> Image:
    img = signwriting_to_image(fsw)
    new_img = Image.new('RGB', (size, size), (255, 255, 255))

    if img.width > size or img.height > size:
        return new_img

    # Calculate the position to paste the image so that it's centered
    x_offset = (size - img.width) // 2
    y_offset = (size - img.height) // 2
    offset = (x_offset, y_offset)

    # Paste the output_im image onto the white background
    new_img.paste(img, offset, img)
    return new_img


class SignWritingCLIPScore(SignWritingMetric):
    def __init__(self, cache_directory=f"{tempfile.gettempdir()}/clip_cache", device=None):
        super().__init__(name="CLIPScore")

        # Init CLIP model
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Init cache
        if cache_directory is None:
            self.cache = {}
            self.cached_texts = set()
        else:
            print("Using cache directory:", cache_directory)
            self.cache = diskcache.Cache(cache_directory, size_limit=2 ** 36)  # 68 GB
            self.cached_texts = set(self.cache.iterkeys())

        # Init device
        self.batch_size = 1
        if device is None:
            if torch.cuda.is_available():
                self.cuda()
        else:
            self.device(device)

    def cuda(self):
        return self.device(torch.device("cuda"))

    def cpu(self):
        return self.device(torch.device("cpu"))

    def device(self, device):
        self.model = self.model.to(device)
        if device.type == 'cuda':
            free_memory, _ = torch.cuda.mem_get_info()
            # max 100 mb per image. We could search for the best batch size, but this is good enough
            self.batch_size = min(128, free_memory // int(100e6))
        else:
            self.batch_size = 16
        return self

    def get_clip_features_batch(self, batch: list[str]):
        images = [signwriting_to_clip_image(text) for text in batch]

        pixels = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.model.device)
        with torch.no_grad():
            img_features = self.model.get_image_features(pixels)
        img_features_normalized = img_features / img_features.norm(p=2, dim=-1, keepdim=True)
        for i, text in enumerate(batch):
            self.cache[text] = img_features_normalized[i].cpu()
            self.cached_texts.add(text)

    def get_clip_features(self, texts: list[str]):
        missing_texts = [text for text in texts if text not in self.cached_texts]

        if len(missing_texts) > 0:
            pbar_disable = len(missing_texts) < self.batch_size
            pbar = tqdm(total=len(texts), initial=len(texts) - len(missing_texts),
                        desc="Computing CLIP features", disable=pbar_disable)

            # pylint: disable=fixme
            # TODO: we could parallelize this if it's too slow for practical use
            batches = (missing_texts[i:i + self.batch_size] for i in range(0, len(missing_texts), self.batch_size))
            for batch in batches:
                self.get_clip_features_batch(batch)
                pbar.update(len(batch))

            pbar.close()

        texts = tqdm(texts, desc="Loading features cache", disable=len(texts) < self.batch_size)
        features = torch.stack([self.cache[text] for text in texts])

        return features.to(self.model.device)

    def score(self, hypothesis: str, reference: str) -> float:
        return self.score_all([hypothesis], [reference])[0][0]

    def score_all(self, hypotheses: list[str], references: list[str]) -> list[list[float]]:
        hyp_features = self.get_clip_features(hypotheses)
        ref_features = self.get_clip_features(references)

        similarities = []
        for hyp_feature in hyp_features:
            hyp_batched = hyp_feature.unsqueeze(0)
            similarity = torch.nn.functional.cosine_similarity(hyp_batched, ref_features)
            similarities.append(similarity.tolist())

        return similarities
