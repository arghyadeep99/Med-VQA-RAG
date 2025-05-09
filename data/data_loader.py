from datasets import load_dataset
import os
import pickle
import json

class DataLoader:
    def __init__(self):
        self.mimic_cxr_image_data = None
        self.mimic_cxr_image_dir = './datasets/mimic-cxr-images-512/'
        self.mimic_cxr_report_dir = './datasets/mimic-cxr/'
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
    
    def create_report_image_mapping(self, sample_size=377110):
        data = []
        patient_groups = sorted(os.listdir(os.path.join(self.mimic_cxr_image_dir,'files'))) # [p10,p11..]
        count = 0
        for patient_group in patient_groups:
            if patient_group != '.DS_Store':
                patient_group_path = os.path.join(self.mimic_cxr_image_dir, 'files', patient_group)
                # print(patient_group_path)
                if not os.path.isdir(patient_group_path):
                    continue

                for patient_folder in sorted(os.listdir(patient_group_path)):
                    patient_folder_path = os.path.join(patient_group_path, patient_folder)
                    if not os.path.isdir(patient_folder_path):
                        continue
                    

                    study_ids = sorted(os.listdir(patient_folder_path))
                    for study_id in study_ids:
                        if study_id != '.DS_Store':
                            study_id_path = os.path.join(patient_folder_path, study_id)
                            file = sorted(os.listdir(study_id_path))[0]
                            if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                                image_path = os.path.join(study_id_path, file)

                                # Corresponding report path
                                report_path = os.path.join(
                                    self.mimic_cxr_report_dir, 'files', patient_group, patient_folder, f"{study_id}.txt"
                                )
                                if not os.path.exists(report_path):
                                    continue

                                entry = {
                                    "id": count,
                                    "report_path": report_path,
                                    "image_path": image_path
                                }
                                data.append(entry)
                                count += 1

                        if count >= sample_size:
                            break
                    if count >= sample_size:
                        break
                if count >= sample_size:
                        break
            
        with open("./image_report_mapping_all_377110.json", "w") as f:
            json.dump(data, f, indent=4)

        print(f"Saved {len(data)} entries to image_report_mapping.json")
            
        

def main():
    loader = DataLoader()
    if os.path.exists(loader.mimic_cxr_pickle_path):
        print("Loading from pickle...")
        loader.load_mimic_cxr_from_pickle()
    else:
        print("Downloading from HF...")
        loader.load_mimic_cxr_from_hf()
        loader.save_mimic_cxr_to_pickle()
    
    # loader.save_mimic_cxr_images_to_jpeg()
    loader.create_report_image_mapping()

if __name__ == "__main__":
    main()