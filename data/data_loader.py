from datasets import load_dataset
import os
import pickle

class DataLoader:
    def __init__(self):
        self.mimic_cxr_image_data = None
        self.mimic_cxr_image_dir = './mimic-cxr-images-512/'
        self.mimic_cxr_pickle_path = 'saved_dataset.pkl'

    def load_mimic_cxr_from_hf(self):
        self.mimic_cxr_image_data = load_dataset('StanfordAIMI/mimic-cxr-images-512', split='train')

    def save_mimic_cxr_to_pickle(self):
        with open(self.mimic_cxr_pickle_path, "wb") as f:
            pickle.dump(self.mimic_cxr_image_data, f)

    def load_mimic_cxr_from_pickle(self):
        with open(self.mimic_cxr_pickle_path, "rb") as f:
            self.mimic_cxr_image_data = pickle.load(f)
        
    def save_mimic_cxr_images_to_jpeg(self):
        for idx, item in enumerate(self.mimic_cxr_image_data):
            image = item['image']
            path = item['path']
            full_path = os.path.join(self.mimic_cxr_image_dir, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            image.save(full_path, format='JPEG', quality=100, subsampling=0)
            if idx % 100 == 0:
                print(f"Saved {idx} images")

        print("----Saving Complete---")

def main():
    loader = DataLoader()
    if os.path.exists(loader.mimic_cxr_pickle_path):
        print("Loading from pickle...")
        loader.load_mimic_cxr_from_pickle()
    else:
        print("Downloading from HF...")
        loader.load_mimic_cxr_from_hf()
        loader.save_mimic_cxr_to_pickle()
    
    loader.save_mimic_cxr_images_to_jpeg()

if __name__ == "__main__":
    main()