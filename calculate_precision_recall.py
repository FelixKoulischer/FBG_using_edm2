# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

##################################################################
#                       Disclaimer
##################################################################

# This code is a slightly modified pytorch equivalent version of the code available at:
#       https://github.com/kynkaat/improved-precision-and-recall-metric/tree/master
# All rights reserved to the respective owners



"""k-NN precision and recall."""

import numpy as np
import torch
from time import time
from tqdm import tqdm
import os
import argparse
from PIL import Image
from torchvision import transforms
import dnnlib
from torch_utils import misc


#----------------------------------------------------------------------------

def batch_pairwise_distances(U, V):
    # ||u - v||^2 = ||u||^2 + ||v||^2 - 2 * u @ v.T
    U_sq = (U ** 2).sum(dim=1, keepdim=True)  # [N, 1]
    V_sq = (V ** 2).sum(dim=1, keepdim=True).T  # [1, M]
    dist = U_sq + V_sq - 2 * U @ V.T
    return dist.clamp(min=0)  # numerical stability

class DistanceBlock(torch.nn.Module):
    """Computes pairwise distances between two batches using multiple GPUs."""
    def __init__(self, num_features, device=None):
        super().__init__()
        self.num_features = num_features
        self.device = 'cuda' 
        self.model = torch.nn.DataParallel(self._DistanceModule())

    class _DistanceModule(torch.nn.Module):
        def forward(self, batch1, batch2):
            return batch_pairwise_distances(batch1, batch2)

    def pairwise_distances(self, U, V):
        """
        U: Tensor of shape [N, D]
        V: Tensor of shape [M, D]
        Returns: Tensor of shape [N, M] with pairwise distances.
        """
        U = torch.tensor(U, dtype=torch.float32).cuda()
        V = torch.tensor(V, dtype=torch.float32).cuda()
        return self.model(U, V).cpu().numpy()
#----------------------------------------------------------------------------

class ManifoldEstimator():
    """Estimates the manifold of given feature vectors."""

    def __init__(self, distance_block, features, row_batch_size=25000, col_batch_size=50000,
                 nhood_sizes=[3], clamp_to_percentile=None, eps=1e-5):
        """Estimate the manifold of given feature vectors.
        
            Args:
                distance_block: DistanceBlock object that distributes pairwise distance
                    calculation to multiple GPUs.
                features (np.array/tf.Tensor): Matrix of feature vectors to estimate their manifold.
                row_batch_size (int): Row batch size to compute pairwise distances
                    (parameter to trade-off between memory usage and performance).
                col_batch_size (int): Column batch size to compute pairwise distances.
                nhood_sizes (list): Number of neighbors used to estimate the manifold.
                clamp_to_percentile (float): Prune hyperspheres that have radius larger than
                    the given percentile.
                eps (float): Small number for numerical stability.
        """
        num_images = features.shape[0]
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.eps = eps
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self._ref_features = features
        self._distance_block = distance_block

        # Estimate manifold of features by calculating distances to k-NN of each sample.
        self.D = np.zeros([num_images, self.num_nhoods], dtype=np.float32)
        distance_batch = np.zeros([row_batch_size, num_images], dtype=np.float32)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)

        for begin1 in range(0, num_images, row_batch_size):
            end1 = min(begin1 + row_batch_size, num_images)
            row_batch = features[begin1:end1]

            for begin2 in range(0, num_images, col_batch_size):
                end2 = min(begin2 + col_batch_size, num_images)
                col_batch = features[begin2:end2]

                # Compute distances between batches.
                distance_batch[0:end1-begin1, begin2:end2] = self._distance_block.pairwise_distances(row_batch, col_batch)
    
            # Find the k-nearest neighbor from the current batch.
            self.D[begin1:end1, :] = np.partition(distance_batch[0:end1-begin1, :], seq, axis=1)[:, self.nhood_sizes]

        if clamp_to_percentile is not None:
            max_distances = np.percentile(self.D, clamp_to_percentile, axis=0)
            self.D[self.D > max_distances] = 0

    def evaluate(self, eval_features, return_realism=False, return_neighbors=False):
        """Evaluate if new feature vectors are at the manifold."""
        num_eval_images = eval_features.shape[0]
        num_ref_images = self.D.shape[0]
        distance_batch = np.zeros([self.row_batch_size, num_ref_images], dtype=np.float32)
        batch_predictions = np.zeros([num_eval_images, self.num_nhoods], dtype=np.int32)
        max_realism_score = np.zeros([num_eval_images,], dtype=np.float32)
        nearest_indices = np.zeros([num_eval_images,], dtype=np.int32)

        for begin1 in range(0, num_eval_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_eval_images)
            feature_batch = eval_features[begin1:end1]

            for begin2 in range(0, num_ref_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_ref_images)
                ref_batch = self._ref_features[begin2:end2]

                distance_batch[0:end1-begin1, begin2:end2] = self._distance_block.pairwise_distances(feature_batch, ref_batch)

            # From the minibatch of new feature vectors, determine if they are in the estimated manifold.
            # If a feature vector is inside a hypersphere of some reference sample, then
            # the new sample lies at the estimated manifold.
            # The radii of the hyperspheres are determined from distances of neighborhood size k.
            samples_in_manifold = distance_batch[0:end1-begin1, :, None] <= self.D
            batch_predictions[begin1:end1] = np.any(samples_in_manifold, axis=1).astype(np.int32)

            max_realism_score[begin1:end1] = np.max(self.D[:, 0] / (distance_batch[0:end1-begin1, :] + self.eps), axis=1)
            nearest_indices[begin1:end1] = np.argmin(distance_batch[0:end1-begin1, :], axis=1)

        if return_realism and return_neighbors:
            return batch_predictions, max_realism_score, nearest_indices
        elif return_realism:
            return batch_predictions, max_realism_score
        elif return_neighbors:
            return batch_predictions, nearest_indices

        return batch_predictions

#----------------------------------------------------------------------------

def knn_precision_recall_features(ref_features, eval_features, nhood_sizes=[3],
                                  row_batch_size=10000, col_batch_size=50000, num_gpus=1):
    """Calculates k-NN precision and recall for two sets of feature vectors.
    
        Args:
            ref_features (np.array/tf.Tensor): Feature vectors of reference images.
            eval_features (np.array/tf.Tensor): Feature vectors of generated images.
            nhood_sizes (list): Number of neighbors used to estimate the manifold.
            row_batch_size (int): Row batch size to compute pairwise distances
                (parameter to trade-off between memory usage and performance).
            col_batch_size (int): Column batch size to compute pairwise distances.
            num_gpus (int): Number of GPUs used to evaluate precision and recall.

        Returns:
            State (dict): Dict that contains precision and recall calculated from
            ref_features and eval_features.
    """
    state = dict()
    num_images = ref_features.shape[0]
    num_features = ref_features.shape[1]

    # Initialize DistanceBlock and ManifoldEstimators.
    distance_block = DistanceBlock(num_features)
    #distance_block = DistanceBlock(num_features, num_gpus)
    ref_manifold  = ManifoldEstimator(distance_block, ref_features, row_batch_size, col_batch_size, nhood_sizes) 
    eval_manifold = ManifoldEstimator(distance_block, eval_features, row_batch_size, col_batch_size, nhood_sizes)

    # Evaluate precision and recall using k-nearest neighbors.
    print('Evaluating k-NN precision and recall with %i samples...' % num_images)
    start = time()

    # Precision: How many points from eval_features are in ref_features manifold.
    precision = ref_manifold.evaluate(eval_features)
    state['precision'] = precision.mean(axis=0)
    print('Precision: ', state['precision'])
    
    # Recall: How many points from ref_features are in eval_features manifold.
    recall = eval_manifold.evaluate(ref_features)
    state['recall'] = recall.mean(axis=0)
    print('Recall: ', state['recall'])

    print('Evaluated k-NN precision and recall in: %gs' % (time() - start))

    return state

#----------------------------------------------------------------------------
# Added DinoV2 model 
class Detector:
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim

    def __call__(self, x): # NCHW, uint8, 3 channels => NC, float32
        raise NotImplementedError # to be overridden by subclass
        
class DINOv2Detector(Detector):
    def __init__(self, resize_mode='torch'):
        super().__init__(feature_dim=1024)
        self.resize_mode = resize_mode
        import warnings
        warnings.filterwarnings('ignore', 'xFormers is not available')
        torch.hub.set_dir(dnnlib.make_cache_dir_path('torch_hub'))
        self.model = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vitl14', trust_repo=True, verbose=False, skip_validation=True)
        self.model.eval().requires_grad_(False)

    def __call__(self, x):
        # Resize images.
        if self.resize_mode == 'pil': # Slow reference implementation that matches the original dgm-eval codebase exactly.
            device = x.device
            x = x.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            x = np.stack([np.uint8(PIL.Image.fromarray(xx, 'RGB').resize((224, 224), PIL.Image.Resampling.BICUBIC)) for xx in x])
            x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
        elif self.resize_mode == 'torch': # Fast practical implementation that yields almost the same results.
            x = torch.nn.functional.interpolate(x.to(torch.float32), size=(224, 224), mode='bicubic', antialias=True)
        else:
            raise ValueError(f'Invalid resize mode "{self.resize_mode}"')

        # Adjust dynamic range.
        x = x.to(torch.float32) / 255
        x = x - misc.const_like(x, [0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        x = x / misc.const_like(x, [0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

        # Run DINOv2 model.
        return self.model.to(x.device)(x)


#----------------------------------------------------------------------------
# Added function to load a set of images as tensors such that these can be pased to the DinoV2 network
def load_images_from_directory(directory, filenames, image_size=224):
    preprocess = transforms.Compose([
        transforms.CenterCrop(min(256, image_size * 2)),  # optional, ImageNet-style
        transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).to(torch.uint8))  # scale to [0, 255] and cast
    ])

    images = []
    for fname in filenames:
        path = os.path.join(directory, fname)
        if not path.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img = Image.open(path).convert('RGB')
        img_tensor = preprocess(img)
        images.append(img_tensor)
    return torch.stack(images)


#----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated_directory', type=str, required=True, help='Directory containing the generated images')
    parser.add_argument('--reference_directory', type=str, required=True, help='Directory containing the reference images')
    parser.add_argument('--num_images', type=int, required=True, help='Number of images to process')
    parser.add_argument('--batch_size', type=int, default = 256, help='Number of images to per batch')
    parser.add_argument('--top_k', type=int, default = 3, help='Number of top-k feature vectors analysed')

    args = parser.parse_args()

    num_images = args.num_images
    batch_size = args.batch_size
    num_batches = num_images//batch_size

    top_k = args.top_k

    gen_image_dir  = args.generated_directory
    print('To analyse directory: ', gen_image_dir)
    real_image_dir = args.reference_directory

    gen_filenames  = sorted(os.listdir(gen_image_dir))[:num_images]  #[f"{i:06d}.png" for i in range(num_images)]
    real_filenames = sorted(os.listdir(real_image_dir))[:num_images] #[f"ILSVRC2012_test_{i:08d}.JPEG" for i in range(num_images)]

    # Load the desired embedding network: DINOV2 or IV3
    embedding_network = DINOv2Detector()

    gen_features_total, real_features_total = [], []
    for batch in tqdm(range(0,num_batches)):
        gen_filenames_batch  = sorted(os.listdir(gen_image_dir))[batch*batch_size: (batch+1)*batch_size]  #[f"{i:06d}.png" for i in range(num_images)]
        real_filenames_batch = sorted(os.listdir(real_image_dir))[batch*batch_size: (batch+1)*batch_size] #[f"ILSVRC2012_test_{i:08d}.JPEG" for i in range(num_images)]
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
        gen_images  = load_images_from_directory(gen_image_dir, gen_filenames_batch).to(device)
        real_images = load_images_from_directory(real_image_dir, real_filenames_batch).to(device)
    
        with torch.no_grad():
            gen_features_batch  = embedding_network(gen_images)
            real_features_batch = embedding_network(real_images)
        gen_features_total.append(gen_features_batch.cpu())
        real_features_total.append(real_features_batch.cpu())
    
    gen_features  = torch.cat(gen_features_total, dim=0)
    real_features = torch.cat(real_features_total, dim=0)
    
    knn_precision_recall_features(real_features.cpu().numpy(), gen_features.cpu().numpy(), nhood_sizes=[top_k])

if __name__ == "__main__":
    main()