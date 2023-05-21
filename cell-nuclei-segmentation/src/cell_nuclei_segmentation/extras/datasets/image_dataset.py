from torch.utils.data import Dataset

from torchvision.transforms import transforms

class ImageDataset(Dataset):
    """
    Stores images, transmutes them and feeds them to the model.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.idx_to_id = {i : id for i, id in enumerate(self.dataset.keys())}

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
        image = self.dataset[image_name]()

        if self.transform:
            image = self.transform(image)

        return image

    def get_image_name(self, idx):
        """
        Returns name of image placed on position 'idx' in the dataset.
        """
        return self.idx_to_id[idx]