import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import re


def extract_numbers(filename):
    match = re.search(r'(\d+)_(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # from pytorch documentation
])


class CEDAR(Dataset):
    def __init__(self, data_dir: str = "CEDAR_dataset", writer_dependent: bool = False):
        self.data_dir = data_dir

        original_path = os.path.join(self.data_dir, "full_org")
        forgery_path = os.path.join(self.data_dir, "full_forg")

        original_signatures = sorted([os.path.join(self.data_dir, "full_org", file) for file in os.listdir(original_path) if file.endswith(".png")], key=extract_numbers)
        assert len(original_signatures) == 1320, f"check length of original signatures (found {len(original_signatures)})"

        forgery_signatures = sorted([os.path.join(self.data_dir, "full_forg", file) for file in os.listdir(forgery_path) if file.endswith(".png")], key=extract_numbers)
        assert len(forgery_signatures) == 1320, f"check length of forgery signatures (found {len(forgery_signatures)})"

        self.data = original_signatures + forgery_signatures
        self.transform = transform

        if writer_dependent:
            self.original_labels = [i for i in range(55) for _ in range(24)]
            self.forgery_labels = [i + max(self.original_labels) + 1 for i in range(55) for _ in range(24)]
        else:
            self.original_labels = [0] * 1320
            self.forgery_labels = [1] * 1320

        self.labels = self.original_labels + self.forgery_labels
        self.classes = len(set(self.labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img, label



class SignatureDataset300(Dataset):
    def __init__(self, data_dir: str = "signature_dataset_300MB", writer_dependent: bool = False):
        self.data_dir = os.path.join(data_dir, "train")

        original_path = os.path.join(self.data_dir, "real")
        forgery_path = os.path.join(self.data_dir, "forg")

        original_signatures = [[] for _ in range(len(os.listdir(original_path)))]
        for i, directory in enumerate(os.listdir(original_path)):
            for file in os.listdir(os.path.join(original_path, directory)):
                original_signatures[i].append(os.path.join(original_path, directory, file))

        forgery_signatures = [[] for _ in range(len(os.listdir(forgery_path)))]
        for i, directory in enumerate(os.listdir(forgery_path)):
            for file in os.listdir(os.path.join(forgery_path, directory)):
                forgery_signatures[i].append(os.path.join(forgery_path, directory, file))

        self.data = [elem for nested_l in original_signatures + forgery_signatures for elem in nested_l]
        self.transform = transform

        # label lengths
        original_lengths = [len(l) for l in original_signatures]
        forgery_lengths = [len(l) for l in forgery_signatures]

        if writer_dependent:
            self.original_labels = [i for i in range(len(original_lengths)) for _ in range(original_lengths[i-1])]
            self.forgery_labels = [i + max(self.original_labels) + 1 for i in range(len(forgery_lengths)) for _ in range(forgery_lengths[i-1])]
            assert len(self.original_labels) == sum(original_lengths)
            assert len(self.forgery_labels) == sum(forgery_lengths)

            assert len(set(self.original_labels)) == len(set(self.forgery_labels)), f"original and forgeries don't match in length (found {len(self.original_labels)}, found {len(self.forgery_labels)})"
        else:
            self.original_labels = [0] * sum(original_lengths)
            self.forgery_labels = [1] * sum(forgery_lengths)

        self.labels = self.original_labels + self.forgery_labels
        self.classes = len(set(self.labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img, label


class BHSigBengali(Dataset):
    def __init__(self, data_dir: str = "BHSig260", writer_dependent: bool = False):
        self.data_dir = os.path.join(data_dir, "Bengali")

        original_signatures = [[] for _ in range(len(os.listdir(self.data_dir)))]
        forgery_signatures = [[] for _ in range(len(os.listdir(self.data_dir)))]
        for i, directory in enumerate(os.listdir(self.data_dir)):
            for file in os.listdir(os.path.join(self.data_dir, directory)):
                if file[-8] == "F":
                    original_signatures[i].append(os.path.join(self.data_dir, directory, file))
                else:
                    forgery_signatures[i].append(os.path.join(self.data_dir, directory, file))

        self.data = [elem for nested_l in original_signatures + forgery_signatures for elem in nested_l]
        self.transform = transform

        # label lengths
        original_lengths = [len(l) for l in original_signatures]
        forgery_lengths = [len(l) for l in forgery_signatures]

        if writer_dependent:
            self.original_labels = [i for i in range(len(original_lengths)) for _ in range(original_lengths[i-1])]
            self.forgery_labels = [i + max(self.original_labels) + 1 for i in range(len(forgery_lengths)) for _ in range(forgery_lengths[i-1])]
            assert len(self.original_labels) == sum(original_lengths)
            assert len(self.forgery_labels) == sum(forgery_lengths)

            assert len(set(self.original_labels)) == len(set(self.forgery_labels)), f"original and forgeries don't match in length (found {len(self.original_labels)}, found {len(self.forgery_labels)})"
        else:
            self.original_labels = [0] * sum(original_lengths)
            self.forgery_labels = [1] * sum(forgery_lengths)

        self.labels = self.original_labels + self.forgery_labels
        self.classes = len(set(self.labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img, label


class BHSigHindi(Dataset):
    def __init__(self, data_dir: str = "BHSig260", writer_dependent: bool = False):
        self.data_dir = os.path.join(data_dir, "Hindi")

        original_signatures = [[] for _ in range(len(os.listdir(self.data_dir)))]
        forgery_signatures = [[] for _ in range(len(os.listdir(self.data_dir)))]
        for i, directory in enumerate(os.listdir(self.data_dir)):
            for file in os.listdir(os.path.join(self.data_dir, directory)):
                if file[-8] == "F":
                    original_signatures[i].append(os.path.join(self.data_dir, directory, file))
                else:
                    forgery_signatures[i].append(os.path.join(self.data_dir, directory, file))

        self.data = [elem for nested_l in original_signatures + forgery_signatures for elem in nested_l]
        self.transform = transform

        # label lengths
        original_lengths = [len(l) for l in original_signatures]
        forgery_lengths = [len(l) for l in forgery_signatures]

        if writer_dependent:
            self.original_labels = [i for i in range(len(original_lengths)) for _ in range(original_lengths[i-1])]
            self.forgery_labels = [i + max(self.original_labels) + 1 for i in range(len(forgery_lengths)) for _ in range(forgery_lengths[i-1])]
            assert len(self.original_labels) == sum(original_lengths)
            assert len(self.forgery_labels) == sum(forgery_lengths)

            assert len(set(self.original_labels)) == len(set(self.forgery_labels)), f"original and forgeries don't match in length (found {len(self.original_labels)}, found {len(self.forgery_labels)})"
        else:
            self.original_labels = [0] * sum(original_lengths)
            self.forgery_labels = [1] * sum(forgery_lengths)

        self.labels = self.original_labels + self.forgery_labels
        self.classes = len(set(self.labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img, label


class CombinedDataset(Dataset):
    def __init__(self, writer_dependent: bool = False):
        cedar = CEDAR(writer_dependent=writer_dependent)
        sig300 = SignatureDataset300(writer_dependent=writer_dependent)
        bhsig_bengali = BHSigBengali(writer_dependent=writer_dependent)
        bhsig_hindi = BHSigHindi(writer_dependent=writer_dependent)

        self.transform = transform
        self.data = cedar.data + sig300.data + bhsig_bengali.data + bhsig_hindi.data

        if writer_dependent:
            self.labels = cedar.labels
            for ds in [sig300, bhsig_bengali, bhsig_hindi]:
                self.labels += [x + max(self.labels) + 1 for x in ds.labels]
            self.classes = len(set(self.labels))
        else:
            self.original_labels = cedar.original_labels + sig300.original_labels + bhsig_bengali.original_labels + bhsig_hindi.original_labels
            self.forgery_labels = cedar.forgery_labels + sig300.forgery_labels + bhsig_bengali.forgery_labels + bhsig_hindi.forgery_labels
            self.labels = self.original_labels + self.forgery_labels
            self.classes = 2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx]).convert('RGB')
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return img, label
