"""
Medical Image Dataset Downloader
Downloads and organizes Kaggle datasets for medical image analysis project
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path
import json
import subprocess
import argparse
from typing import Dict, List

class KaggleDatasetDownloader:
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            'chest_xray': {
                'kaggle_id': 'paultimothymooney/chest-xray-pneumonia',
                'download_dir': self.data_dir / 'chest_xray',
                'description': 'Chest X-Ray Images (Pneumonia) - 5,863 images'
            },
            'skin_lesion': {
                'kaggle_id': 'kmader/skin-cancer-mnist-ham10000',
                'download_dir': self.data_dir / 'skin_lesion',
                'description': 'Skin Cancer MNIST: HAM10000 - 10,015 images'
            },
            'brain_tumor': {
                'kaggle_id': 'masoudnickparvar/brain-tumor-mri-dataset',
                'download_dir': self.data_dir / 'brain_tumor',
                'description': 'Brain Tumor Classification (MRI) - 3,264 images'
            }
        }
        
    def check_kaggle_credentials(self) -> bool:
        kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
        
        if not kaggle_json.exists():
            print("❌ Kaggle API credentials not found!")
            print("\nPlease follow these steps:")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Scroll to 'API' section")
            print("3. Click 'Create New API Token'")
            print("4. Save kaggle.json to ~/.kaggle/kaggle.json")
            print("5. Run: chmod 600 ~/.kaggle/kaggle.json (on Linux/Mac)")
            return False
        
        # Verify permissions (Unix-based systems)
        if os.name != 'nt':  # Not Windows
            stats = os.stat(kaggle_json)
            if stats.st_mode & 0o077:
                print("⚠️  Warning: kaggle.json has incorrect permissions")
                print("Run: chmod 600 ~/.kaggle/kaggle.json")
        
        return True
    
    def install_kaggle_api(self) -> bool:
        try:
            import kaggle
            print("✓ Kaggle API already installed")
            return True
        except ImportError:
            print("📦 Installing Kaggle API...")
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', 'kaggle', '-q'
                ])
                print("✓ Kaggle API installed successfully")
                return True
            except subprocess.CalledProcessError:
                print("❌ Failed to install Kaggle API")
                return False
    
    def download_dataset(self, dataset_key: str, force: bool = False) -> bool:
        if dataset_key not in self.datasets:
            print(f"❌ Unknown dataset: {dataset_key}")
            return False
        
        dataset = self.datasets[dataset_key]
        download_dir = dataset['download_dir']
        
        # Check if already downloaded
        if download_dir.exists() and not force:
            print(f"✓ Dataset '{dataset_key}' already exists at {download_dir}")
            response = input("Re-download? (y/n): ").strip().lower()
            if response != 'y':
                return True
        
        # Create download directory
        download_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📥 Downloading {dataset['description']}...")
        print(f"   Kaggle ID: {dataset['kaggle_id']}")
        
        try:
            # Import kaggle after installation
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            # Initialize Kaggle API
            api = KaggleApi()
            api.authenticate()
            
            # Download dataset
            api.dataset_download_files(
                dataset['kaggle_id'],
                path=download_dir,
                unzip=True,
                quiet=False
            )
            
            print(f"✓ Successfully downloaded to {download_dir}")
            
            # Clean up zip files if any
            for zip_file in download_dir.glob('*.zip'):
                zip_file.unlink()
            
            return True
            
        except Exception as e:
            print(f"❌ Error downloading {dataset_key}: {str(e)}")
            return False
    
    def download_all(self, force: bool = False) -> Dict[str, bool]:
        results = {}
        
        for dataset_key in self.datasets:
            success = self.download_dataset(dataset_key, force)
            results[dataset_key] = success
        
        return results
    
    def verify_datasets(self) -> Dict[str, Dict]:
        info = {}
        
        for dataset_key, dataset in self.datasets.items():
            download_dir = dataset['download_dir']
            
            if not download_dir.exists():
                info[dataset_key] = {
                    'status': 'not_downloaded',
                    'path': str(download_dir),
                    'image_count': 0
                }
                continue

            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm'}
            image_count = sum(
                1 for f in download_dir.rglob('*') 
                if f.suffix.lower() in image_extensions
            )
            total_size = sum(
                f.stat().st_size for f in download_dir.rglob('*') if f.is_file()
            )
            size_mb = total_size / (1024 * 1024)
            
            info[dataset_key] = {
                'status': 'downloaded',
                'path': str(download_dir),
                'image_count': image_count,
                'size_mb': round(size_mb, 2),
                'subdirectories': [d.name for d in download_dir.iterdir() if d.is_dir()]
            }
        
        return info
    
    def organize_chest_xray(self):
        """
        Organize chest X-ray dataset into proper structure
        """
        chest_dir = self.data_dir / 'chest_xray'
        
        if (chest_dir / 'chest_xray').exists():
            # Sometimes Kaggle creates nested directory
            nested = chest_dir / 'chest_xray'
            for item in nested.iterdir():
                shutil.move(str(item), str(chest_dir))
            nested.rmdir()
        
        print(f"✓ Chest X-ray dataset organized at {chest_dir}")
    
    def organize_skin_lesion(self):
        """
        Organize skin lesion dataset into proper structure
        """
        skin_dir = self.data_dir / 'skin_lesion'
        print(f"✓ Skin lesion dataset at {skin_dir}")
    
    def organize_brain_tumor(self):
        """
        Organize brain tumor dataset into proper structure
        """
        brain_dir = self.data_dir / 'brain_tumor'
        print(f"✓ Brain tumor dataset at {brain_dir}")
    
    def organize_all_datasets(self):
        """
        Organize all downloaded datasets
        """
        print("\n📁 Organizing datasets...")
        
        if (self.data_dir / 'chest_xray').exists():
            self.organize_chest_xray()
        
        if (self.data_dir / 'skin_lesion').exists():
            self.organize_skin_lesion()
        
        if (self.data_dir / 'brain_tumor').exists():
            self.organize_brain_tumor()
    
    def create_dataset_info(self):
        """
        Create a JSON file with dataset information
        """
        info = self.verify_datasets()
        
        info_file = self.data_dir / 'dataset_info.json'
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n📄 Dataset information saved to {info_file}")
        return info
    
    def print_summary(self):
        """
        Print summary of downloaded datasets
        """
        info = self.verify_datasets()
        
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        
        for dataset_key, data in info.items():
            print(f"\n📊 {dataset_key.upper().replace('_', ' ')}")
            print(f"   Status: {data['status']}")
            
            if data['status'] == 'downloaded':
                print(f"   Images: {data['image_count']:,}")
                print(f"   Size: {data['size_mb']:.2f} MB")
                print(f"   Path: {data['path']}")
                if data.get('subdirectories'):
                    print(f"   Subdirectories: {', '.join(data['subdirectories'][:5])}")
        
        print("\n" + "="*60)


def main():
    """
    Main function to run the dataset downloader
    """
    parser = argparse.ArgumentParser(
        description='Download Kaggle datasets for medical image analysis'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['chest_xray', 'skin_lesion', 'brain_tumor', 'all'],
        default='all',
        help='Specific dataset to download (default: all)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if dataset exists'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Directory to store datasets (default: data/raw)'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing datasets without downloading'
    )
    
    args = parser.parse_args()
    downloader = KaggleDatasetDownloader(data_dir=args.data_dir)
    
    print("🏥 Medical Image Dataset Downloader")
    print("="*60)
    
    # Verify only mode
    if args.verify_only:
        downloader.print_summary()
        downloader.create_dataset_info()
        return
    
    # Check Kaggle credentials
    if not downloader.check_kaggle_credentials():
        return
    
    # Install Kaggle API if needed
    if not downloader.install_kaggle_api():
        return
    if args.dataset == 'all':
        print("\n📥 Downloading all datasets...")
        results = downloader.download_all(force=args.force)
        print("\n" + "="*60)
        print("DOWNLOAD RESULTS")
        print("="*60)
        for dataset, success in results.items():
            status = "✓ Success" if success else "✗ Failed"
            print(f"{dataset}: {status}")
    else:
        downloader.download_dataset(args.dataset, force=args.force)
    downloader.organize_all_datasets()
    downloader.print_summary()
    downloader.create_dataset_info()
    
    print("\n✅ Download process completed!")
    print("\nNext steps:")
    print("1. Explore the data in notebooks/01_data_exploration.ipynb")
    print("2. Run preprocessing on the images")
    print("3. Start training your CNN models")


if __name__ == "__main__":
    main()
