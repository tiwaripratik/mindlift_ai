# -*- coding: utf-8 -*-
"""
Helper script to verify and prepare model ZIP files for Colab upload

This script helps you:
1. Verify ZIP files are valid before uploading
2. Create proper ZIP archives from model folders
3. Check file integrity

Usage:
    python verify_and_prepare_models.py
    
    # Verify existing ZIP files:
    python verify_and_prepare_models.py --verify "D:\Downloads\detect.zip"
    
    # Create ZIP from folder:
    python verify_and_prepare_models.py --create-zip "D:\path\to\detect" "D:\Downloads\detect.zip"
"""

import os
import sys
import zipfile
import argparse
from pathlib import Path


def verify_zip_file(zip_path: str) -> dict:
    """Verify if a ZIP file is valid and get information about it"""
    result = {
        "valid": False,
        "file_exists": False,
        "file_size": 0,
        "file_count": 0,
        "files": [],
        "error": None
    }
    
    zip_path = Path(zip_path)
    
    if not zip_path.exists():
        result["error"] = f"File not found: {zip_path}"
        return result
    
    result["file_exists"] = True
    result["file_size"] = zip_path.stat().st_size
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Test integrity
            bad_file = zf.testzip()
            if bad_file:
                result["error"] = f"Corrupted file in archive: {bad_file}"
                return result
            
            # Get file list
            result["files"] = zf.namelist()
            result["file_count"] = len(result["files"])
            result["valid"] = True
            
    except zipfile.BadZipFile as e:
        result["error"] = f"Invalid ZIP file: {e}"
    except Exception as e:
        result["error"] = f"Error reading ZIP: {e}"
    
    return result


def create_zip_from_folder(folder_path: str, output_zip: str) -> bool:
    """Create a ZIP archive from a folder"""
    folder_path = Path(folder_path)
    output_zip = Path(output_zip)
    
    if not folder_path.exists():
        print(f"❌ Folder not found: {folder_path}")
        return False
    
    if not folder_path.is_dir():
        print(f"❌ Path is not a directory: {folder_path}")
        return False
    
    print(f"Creating ZIP archive from: {folder_path}")
    print(f"Output: {output_zip}")
    
    try:
        import shutil
        shutil.make_archive(str(output_zip.with_suffix('')), 'zip', str(folder_path))
        print(f"✓ Successfully created: {output_zip}")
        
        # Verify the created ZIP
        result = verify_zip_file(output_zip)
        if result["valid"]:
            print(f"✓ Verified: ZIP contains {result['file_count']} files")
            return True
        else:
            print(f"⚠️ Warning: Created ZIP but verification failed: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating ZIP: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify and prepare model ZIP files for Colab"
    )
    parser.add_argument(
        '--verify',
        type=str,
        help='Verify a ZIP file (provide path)'
    )
    parser.add_argument(
        '--create-zip',
        nargs=2,
        metavar=('FOLDER', 'OUTPUT_ZIP'),
        help='Create ZIP from folder: --create-zip folder_path output.zip'
    )
    parser.add_argument(
        '--check-models',
        action='store_true',
        help='Check common model locations for detect and empathy folders'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Model ZIP Verification and Preparation Tool")
    print("=" * 80)
    
    if args.verify:
        print(f"\nVerifying ZIP file: {args.verify}")
        print("-" * 80)
        
        result = verify_zip_file(args.verify)
        
        if result["file_exists"]:
            size_mb = result["file_size"] / (1024 * 1024)
            print(f"File Size: {size_mb:.2f} MB ({result['file_size']:,} bytes)")
        
        if result["valid"]:
            print(f"✓ ZIP file is VALID")
            print(f"  Files in archive: {result['file_count']}")
            print(f"\nFirst 10 files:")
            for i, file_name in enumerate(result["files"][:10], 1):
                print(f"  {i}. {file_name}")
            if result["file_count"] > 10:
                print(f"  ... and {result['file_count'] - 10} more files")
            
            # Check if it looks like a model directory
            has_adapter_config = any("adapter_config.json" in f for f in result["files"])
            has_tokenizer = any("tokenizer" in f.lower() for f in result["files"])
            
            if has_adapter_config or has_tokenizer:
                print(f"\n✓ This appears to be a valid model archive!")
                print(f"  - Contains adapter config: {has_adapter_config}")
                print(f"  - Contains tokenizer files: {has_tokenizer}")
            else:
                print(f"\n⚠️ Warning: Doesn't look like a model archive")
                print(f"  (No adapter_config.json or tokenizer files found)")
        else:
            print(f"❌ ZIP file is INVALID or CORRUPTED")
            print(f"  Error: {result['error']}")
            print(f"\nSuggestions:")
            print(f"  1. Try opening the file with WinRAR or 7-Zip")
            print(f"  2. Re-download or re-create the ZIP file")
            print(f"  3. If you have the folder, use --create-zip to make a new ZIP")
    
    elif args.create_zip:
        folder_path, output_zip = args.create_zip
        print(f"\nCreating ZIP archive:")
        print(f"  From: {folder_path}")
        print(f"  To: {output_zip}")
        print("-" * 80)
        
        success = create_zip_from_folder(folder_path, output_zip)
        
        if success:
            print(f"\n✓ ZIP file ready for upload!")
            print(f"\nNext steps for Colab:")
            print(f"  1. Upload {output_zip} to Colab")
            print(f"  2. Run: !python intelligent_conversation_system.py --interactive")
        else:
            print(f"\n❌ Failed to create ZIP file")
    
    elif args.check_models:
        print(f"\nChecking for model folders in common locations...")
        print("-" * 80)
        
        common_locations = [
            "./detect",
            "./empathy",
            "../detect",
            "../empathy",
            "D:/detect",
            "D:/empathy",
            "D:/Downloads/detect",
            "D:/Downloads/empathy",
        ]
        
        found_models = []
        for loc in common_locations:
            path = Path(loc)
            if path.exists() and path.is_dir():
                adapter_config = path / "adapter_config.json"
                if adapter_config.exists():
                    found_models.append(str(path))
                    print(f"✓ Found model: {path}")
        
        if found_models:
            print(f"\nFound {len(found_models)} model folder(s):")
            for model_path in found_models:
                print(f"  - {model_path}")
            print(f"\nTo create ZIP files, run:")
            for model_path in found_models:
                model_name = Path(model_path).name
                print(f"  python verify_and_prepare_models.py --create-zip \"{model_path}\" \"D:\\Downloads\\{model_name}.zip\"")
        else:
            print(f"⚠️ No model folders found in common locations")
            print(f"\nPlease specify the exact path to your model folders")
    
    else:
        # Interactive mode
        print("\nNo arguments provided. Checking for model files...")
        print("\nTo verify a ZIP file:")
        print("  python verify_and_prepare_models.py --verify \"D:\\Downloads\\detect.zip\"")
        print("\nTo create ZIP from folder:")
        print("  python verify_and_prepare_models.py --create-zip \"D:\\path\\to\\detect\" \"D:\\Downloads\\detect.zip\"")
        print("\nTo check for model folders:")
        print("  python verify_and_prepare_models.py --check-models")
        
        # Check if detect.zip exists in Downloads
        downloads_detect = Path("D:/Downloads/detect.zip")
        downloads_empathy = Path("D:/Downloads/empathy.zip")
        
        if downloads_detect.exists():
            print(f"\nFound detect.zip in Downloads, verifying...")
            result = verify_zip_file(downloads_detect)
            if result["valid"]:
                print(f"✓ detect.zip is valid ({result['file_count']} files)")
            else:
                print(f"❌ detect.zip is corrupted: {result['error']}")
        
        if downloads_empathy.exists():
            print(f"\nFound empathy.zip in Downloads, verifying...")
            result = verify_zip_file(downloads_empathy)
            if result["valid"]:
                print(f"✓ empathy.zip is valid ({result['file_count']} files)")
            else:
                print(f"❌ empathy.zip is corrupted: {result['error']}")


if __name__ == "__main__":
    main()

