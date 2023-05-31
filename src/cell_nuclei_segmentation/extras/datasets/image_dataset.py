from torch.utils.data import Dataset

from torchvision.transforms import transforms

class ImageDataset(Dataset):
    """
    Stores images, transmutes them and feeds them to the model.
    """
    def __init__(self, raw_images, target_masks):
        self.images = raw_images
        self.targets = target_masks
        self.idx_to_id = {i : id for i, id in enumerate(self.images.keys())}
        self.idx_to_id_target = {i : id for i, id in enumerate(self.targets.keys())}

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        """
        Returns length of dataset.
        """
        return len(self.idx_to_id)

    def __getitem__(self, idx):
        """
        Returns image placed on position 'idx' in the dataset and image's classification label.
        """
        image_name = self.get_image_name(idx)
        image = self.images[image_name]()
        target = self.targets[image_name]()

        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        return image, target

    def get_image_name(self, idx):
        """
        Returns name of image placed on position 'idx' in the dataset.
        """
        return self.idx_to_id[idx]
    
    def get_target_name(self, idx):
        """
        Returns name of image placed on position 'idx' in the dataset.
        """
        return self.idx_to_id_target[idx]