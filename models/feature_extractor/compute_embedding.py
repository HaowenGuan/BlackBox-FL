import argparse
import os
import torch
import torchvision
import tqdm
import numpy as np

from dataset.wikiart_dataset import WikiArt
from dataset.fungi_dataset import FungiDataset
from dataset.coco_dataset import CocoDataset
from models.feature_extractor.pretrained_fe import get_fe_metadata


"""
See compute_embeddings.sh


python src/scripts/compute_embeddings.py \
     --detailed_name \
     --fe_type timm:vit_base_patch16_clip_224.openai:768 \
     --batch_size 1024 \
     --gpu 0 \
     --model ICL \
     --image_embedding_cache_dir ../latest_imagenet/cached_embeddings/ \
     --dataset wikiart-style
"""

parser = argparse.ArgumentParser()
parser.add_argument('--detailed_name', action='store_true', default=True)
parser.add_argument('--fe_type', type=str, default='timm:vit_base_patch16_clip_224.openai:768')
parser.add_argument('--fe_dtype', type=str, default='float32')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--model', type=str, default='ICL')
parser.add_argument('--image_embedding_cache_dir', type=str, default='../latest_imagenet/cached_embeddings/')
parser.add_argument('--dataset', type=str, default='wikiart-style')
args = parser.parse_args()


# Figure out what feature extractor we're using and get associated metadata.
fe_metadata = get_fe_metadata(args)
transforms = fe_metadata['test_transform']
fe_model = fe_metadata['fe']
device = torch.device(f'cuda:{args.gpu}')
fe_model = fe_model.to(device)
batch_size = 1


def _get_dataloader(dataset: str, split: str, batch_size, transforms):
  if dataset == 'imagenet':
    data = torchvision.datasets.ImageNet('../image_datasets/latest_imagenet', split=split, transform=transforms)
  elif dataset == 'wikiart-style':
    data = WikiArt(split=split, class_column='style', transform=transforms)
  elif dataset == 'wikiart-artist':
    data = WikiArt(split=split, class_column='artist', transform=transforms)
  elif dataset == 'wikiart-genre':
    data = WikiArt(split=split, class_column='genre', transform=transforms)
  elif dataset == 'fungi':
    data = FungiDataset(split=split, transform=transforms)
  elif dataset == 'coco':
    data = CocoDataset(split=split, transform=transforms)
  dataloader = torch.utils.data.DataLoader(
    data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=12)
  return dataloader


for split in ['train', 'val']:
  # Load the dataset.
  dataloader = _get_dataloader(args.dataset, split, batch_size, transforms)

  cache_dir = os.path.join(args.image_embedding_cache_dir, args.fe_type, split)
  print('Writing embeddings to cache dir', cache_dir)

  # For computing average embeddings
  running_average = None
  examples_seen = 0

  for batch_idx, (images, labels) in tqdm.tqdm(enumerate(dataloader)):
    images = images.to(device)
    with torch.no_grad():
      embeddings = fe_model(images)

      # Update the average
      b = embeddings.shape[0]
      batch_mean = torch.mean(embeddings, dim=0)
      if running_average is None:
        running_average = batch_mean
      else:
        running_average = running_average * (
                examples_seen /
                (examples_seen + b)) + batch_mean * (b / examples_seen)
      examples_seen += b

      embeddings = embeddings.to('cpu').numpy()

      # Now write out the embeddings for each image individually
      for i in range(embeddings.shape[0]):
        embedding = embeddings[i, :].reshape(-1)
        cls = labels[i].reshape(-1).item()
        cls_dir = os.path.join(cache_dir, str(cls))
        if not os.path.exists(cls_dir):
          os.makedirs(cls_dir)
        example_idx = batch_idx * batch_size + i
        filename = os.path.join(cls_dir, f'{example_idx}.npy')
        np.save(filename, embedding)

  # Now write out the average
  running_average = running_average.to('cpu').numpy()
  average_filename = os.path.join(cache_dir, f'split_average.npy')
  np.save(average_filename, running_average)

if __name__ == '__main__':
    pass