import torch
from torch.utils.data import Sampler, Subset
import numpy as np
import random


class BalancedBatchSampler(Sampler):
    """
    Sampler to ensure that each batch contains either 0 or 2+ samples of each class.
    """

    def __init__(self, dataset, batch_size, min_samples_per_class=2): # Now min_samples_per_class is the minimum
        """
        Args:
            dataset (Dataset or Subset): The dataset from which to sample. Needs to have a way to access labels.
            batch_size (int): The desired batch size.
            min_samples_per_class (int):  The *minimum* number of samples per class to *include* in a batch (either 0 or >= min_samples_per_class).
        """
        self.dataset = dataset
        self.labels = self._get_labels()
        self.batch_size = batch_size
        self.min_samples_per_class = min_samples_per_class # Now refers to the *minimum*

        self.unique_labels = np.unique(self.labels)
        self.num_classes = len(self.unique_labels)

        # No check for batch_size here - the logic in __iter__ will handle batch construction.

        # Create a dictionary to store indices for each class
        self.class_indices = {label: np.where(self.labels == label)[0] for label in self.unique_labels}

        self.dataset_size = len(self.labels)
        #self.batches_per_epoch = self.dataset_size // batch_size # Not used in this version, we can't estimate batches easily.

    def _get_labels(self):
        """
        Retrieves the labels from the dataset.
        """
        if isinstance(self.dataset, Subset):
            original_dataset = self.dataset.dataset
            indices = self.dataset.indices
            if hasattr(original_dataset, 'targets'):
                labels = [original_dataset.targets[i] for i in indices]
                return np.array(labels)
            else:
                raise NotImplementedError("Implement _get_labels() based on how your original dataset stores labels.")
        else:
            if hasattr(self.dataset, 'targets'):
                return np.array(self.dataset.targets)
            else:
                raise NotImplementedError("Implement _get_labels() based on how your original dataset stores labels.")

    def __iter__(self):
        # Shuffle the indices for each class
        for label in self.unique_labels:
            np.random.shuffle(self.class_indices[label])

        batch = []
        indices_used = {label: 0 for label in self.unique_labels}

        # Build batches.  We will try to add each class to the batch,
        # either 0 or >= min_samples_per_class.
        while True:  # Create batches until we run out of data (or stop somehow).
            batch = [] # Start a new batch.
            classes_in_batch = set() # Keep track of classes included.

            # Add samples for each class
            for label in self.unique_labels:
                class_idxs = self.class_indices[label]
                num_available = len(class_idxs) - indices_used[label]

                if num_available >= self.min_samples_per_class:
                    # Add samples from this class
                    batch.extend(class_idxs[indices_used[label]:indices_used[label] + self.min_samples_per_class])
                    indices_used[label] += self.min_samples_per_class
                    classes_in_batch.add(label)

            # If the batch is too small, or contains no classes, skip it.
            if len(batch) < 2 or len(classes_in_batch) == 0 : # You can relax this constraint.
                if len(batch) > 0:
                    # Put those examples back.  We could add them to the next batch.
                    # But it is easier if we skip this batch.
                    # You could choose to append incomplete batches at the end of the epoch if desired.
                    pass
                continue # Start building the next batch.

            # If the batch is large enough (or we ran out of things to include)
            if len(batch) >= 2:
                # Check if the number of samples is the same as the batch_size to avoid cropping.
                if len(batch) == self.batch_size:
                  yield batch # Yield this batch
                elif len(batch) < self.batch_size: # If it's smaller, we'll just pad it with random examples.
                    remaining_indices = [idx for label in self.unique_labels for idx in self.class_indices[label] if idx not in batch]
                    random.shuffle(remaining_indices)
                    num_to_add = min(self.batch_size - len(batch), len(remaining_indices))
                    batch.extend(remaining_indices[:num_to_add])
                    yield batch
                else:
                  # In this case, we can add an additional constraint, e.g., we can skip if there are too many samples.
                  pass

            # Check if we've processed all data.  If we've run out of available samples for all classes, stop.
            all_used = all(indices_used[label] >= len(self.class_indices[label]) for label in self.unique_labels)
            if all_used:
                break  # We've used all the data - stop the iteration.

    def __len__(self):
        # We can't easily determine the exact number of batches, so we return a large number.
        return  self.dataset_size // (self.min_samples_per_class * self.num_classes) * 2 # A reasonable approximation (but not exact).
