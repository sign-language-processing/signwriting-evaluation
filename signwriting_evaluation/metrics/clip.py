import hashlib
import tempfile
from typing import Union

import diskcache
import torch
from PIL import Image
from signwriting.visualizer.visualize import signwriting_to_image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from signwriting_evaluation.metrics.base import SignWritingMetric

CLIPInput = Union[str, Image.Image]


def signwriting_to_clip_image(signwriting: CLIPInput, size=224) -> Image:
    new_img = Image.new('RGB', (size, size), (255, 255, 255))

    if isinstance(signwriting, str):
        try:
            img = signwriting_to_image(signwriting, trust_box=False)
        except ValueError as value_error:
            # This may happen when the M box maximum values are lower
            # than the symbols minimum values
            print(value_error)
            return new_img
    else:
        img = signwriting

    if img.width > size or img.height > size:
        return new_img

    # Calculate the position to paste the image so that it's centered
    x_offset = (size - img.width) // 2
    y_offset = (size - img.height) // 2
    offset = (x_offset, y_offset)

    # Paste the output_im image onto the white background
    if img.mode == 'RGBA':
        new_img.paste(img, offset, img)
    else:
        new_img.paste(img, offset)
    return new_img


class SignWritingCLIPScore(SignWritingMetric):
    def __init__(self,
                 cache_directory=f"{tempfile.gettempdir()}/clip_cache",
                 model_id="openai/clip-vit-base-patch32",
                 device=None):
        super().__init__(name="CLIPScore")

        # Init CLIP model
        self.model = AutoModel.from_pretrained(model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)

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

    def get_clip_features_batch(self, batch: list[CLIPInput]):
        images = [signwriting_to_clip_image(item) for item in batch]

        pixels = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.model.device)
        with torch.no_grad():
            img_features = self.model.get_image_features(pixels)
        img_features_normalized = img_features / img_features.norm(p=2, dim=-1, keepdim=True)
        for i, item in enumerate(batch):
            cache_name = self.cache_name(item)
            self.cache[cache_name] = img_features_normalized[i].cpu()
            self.cached_texts.add(cache_name)

    def cache_name(self, clip_input: CLIPInput):
        if isinstance(clip_input, Image.Image):
            return hashlib.md5(clip_input.tobytes()).hexdigest()
        return clip_input

    def get_clip_features(self, inputs: list[CLIPInput], progress_bar=True):
        missing = [clip_input for clip_input in inputs if self.cache_name(clip_input) not in self.cached_texts]

        if len(missing) > 0:
            pbar_disable = not progress_bar or len(missing) <= self.batch_size
            pbar = tqdm(total=len(inputs), initial=len(inputs) - len(missing),
                        desc="Computing CLIP features", disable=pbar_disable)

            # pylint: disable=fixme
            # TODO: we could parallelize this if it's too slow for practical use
            batches = (missing[i:i + self.batch_size] for i in range(0, len(missing), self.batch_size))
            for batch in batches:
                self.get_clip_features_batch(batch)
                pbar.update(len(batch))

            pbar.close()

        texts = tqdm(inputs, desc="Loading features cache",
                     disable=not progress_bar or len(inputs) <= self.batch_size)
        cached_features = [self.cache[self.cache_name(text)].cpu() for text in texts]
        features = torch.stack(cached_features)

        return features.to(self.model.device)

    def score(self, hypothesis: CLIPInput, reference: CLIPInput) -> float:
        return self.score_all([hypothesis], [reference])[0][0]

    def score_all(self, hypotheses: list[CLIPInput], references: list[CLIPInput],
                  progress_bar=True) -> list[list[float]]:
        hyp_features = self.get_clip_features(hypotheses, progress_bar)
        ref_features = self.get_clip_features(references, progress_bar)

        similarities = []
        for hyp_feature in hyp_features:
            hyp_batched = hyp_feature.unsqueeze(0)
            # pylint: disable=not-callable
            similarity = torch.nn.functional.cosine_similarity(hyp_batched, ref_features)
            similarities.append(similarity.tolist())

        return similarities
