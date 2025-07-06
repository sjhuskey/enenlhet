import dropbox
import os
import requests
from requests.auth import HTTPBasicAuth

# 1. Dropbox Setup
DROPBOX_ACCESS_TOKEN = 'insert_your_access_token_here'  # paste manually each session

def initialize_dropbox(access_token):
    """Initialize Dropbox client with access token"""
    global dbx
    dbx = dropbox.Dropbox(access_token)
    try:
        # Check if the authentication works
        account = dbx.users_get_current_account()
        print(f"✅ Dropbox authenticated successfully for: {account.name.display_name}")
        return True
    except Exception as e:
        print(f"❌ Dropbox authentication failed: {e}")
        return False


# Function to download a file from Dropbox and save it locally in Colab
def download_from_dropbox(dropbox_file_path, local_file_path):
    """Download a file from Dropbox to local path"""
    try:
        print(f"📥 Downloading {dropbox_file_path} to {local_file_path}")
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        with open(local_file_path, "wb") as f:
            metadata, res = dbx.files_download(path=dropbox_file_path)
            f.write(res.content)
        
        print(f"✅ Successfully downloaded {dropbox_file_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {dropbox_file_path}: {e}")
        return False

def download_dataset_from_dropbox(dropbox_dataset_path, local_dataset_path):
    """Download the dataset file from Dropbox"""
    return download_from_dropbox(dropbox_dataset_path, local_dataset_path)

def download_directory_from_dropbox(dropbox_dir_path, local_dir_path):
    """Recursively download an entire directory from Dropbox to local path"""
    try:
        print(f"📁 Starting recursive download from {dropbox_dir_path} to {local_dir_path}")
        
        # Create local directory if it doesn't exist
        os.makedirs(local_dir_path, exist_ok=True)
        
        # List all files and folders in the Dropbox directory
        result = dbx.files_list_folder(dropbox_dir_path, recursive=True)
        entries = result.entries
        
        # Handle pagination if there are more entries
        while result.has_more:
            result = dbx.files_list_folder_continue(result.cursor)
            entries.extend(result.entries)
        
        print(f"📋 Found {len(entries)} items to download")
        
        # Download each file
        file_count = 0
        for entry in entries:
            if isinstance(entry, dropbox.files.FileMetadata):
                # It's a file, download it
                relative_path = entry.path_lower.replace(dropbox_dir_path.lower(), '').lstrip('/')
                local_file_path = os.path.join(local_dir_path, relative_path)
                
                # Create directory structure if needed
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                
                # Download the file
                try:
                    with open(local_file_path, "wb") as f:
                        metadata, res = dbx.files_download(path=entry.path_lower)
                        f.write(res.content)
                    
                    file_count += 1
                    print(f"✅ Downloaded ({file_count}/{len([e for e in entries if isinstance(e, dropbox.files.FileMetadata)])}): {relative_path}")
                except Exception as e:
                    print(f"❌ Failed to download {entry.path_lower}: {e}")
            
            elif isinstance(entry, dropbox.files.FolderMetadata):
                # It's a folder, create the local directory
                relative_path = entry.path_lower.replace(dropbox_dir_path.lower(), '').lstrip('/')
                local_folder_path = os.path.join(local_dir_path, relative_path)
                os.makedirs(local_folder_path, exist_ok=True)
        
        print(f"🎉 Successfully downloaded {file_count} files from {dropbox_dir_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download directory {dropbox_dir_path}: {e}")
        return False

# Function to upload a single file to Dropbox
def upload_to_dropbox(local_file_path, dropbox_file_path):
    """Upload a single file to Dropbox with automatic chunked upload for large files."""
    file_size = os.path.getsize(local_file_path)
    
    if file_size > 150 * 1024 * 1024:  # 150 MB limit for standard upload
        print(f"File {local_file_path} is large ({file_size/(1024*1024):.1f} MB), using chunked upload.")
        upload_large_file_to_dropbox(local_file_path, dropbox_file_path)
    else:
        print(f"Uploading {local_file_path} ({file_size/(1024*1024):.1f} MB)")
        try:
            with open(local_file_path, "rb") as f:
                dbx.files_upload(f.read(), dropbox_file_path, mode=dropbox.files.WriteMode.overwrite)
            print(f"✅ Successfully uploaded to {dropbox_file_path}")
        except Exception as e:
            print(f"❌ Failed to upload {local_file_path}: {e}")

# Function to handle chunked upload for large files
def upload_large_file_to_dropbox(local_file_path, dropbox_file_path, chunk_size=4 * 1024 * 1024):
    """Uploads large files in chunks to Dropbox (supports files larger than 150 MB)."""
    file_size = os.path.getsize(local_file_path)
    with open(local_file_path, 'rb') as f:
        if file_size <= chunk_size:
            print(f"Uploading {local_file_path} directly (size: {file_size} bytes)")
            dbx.files_upload(f.read(), dropbox_file_path)
        else:
            print(f"Uploading {local_file_path} in chunks (size: {file_size} bytes)")
            upload_session_start_result = dbx.files_upload_session_start(f.read(chunk_size))
            cursor = dropbox.files.UploadSessionCursor(session_id=upload_session_start_result.session_id,
                                                       offset=f.tell())
            commit = dropbox.files.CommitInfo(path=dropbox_file_path, mode=dropbox.files.WriteMode.overwrite)

            # Upload chunks
            while f.tell() < file_size:
                if ((file_size - f.tell()) <= chunk_size):
                    print("Finishing upload...")
                    dbx.files_upload_session_finish(f.read(chunk_size), cursor, commit)
                else:
                    dbx.files_upload_session_append_v2(f.read(chunk_size), cursor)
                    cursor.offset = f.tell()

# Function to upload all files in a directory (handles large files)
def upload_directory_to_dropbox(local_dir_path, dropbox_dir_path):
    """Uploads all files in a local directory to Dropbox, handling large files with chunked upload."""
    for root, dirs, files in os.walk(local_dir_path):
        for filename in files:
            local_file_path = os.path.join(root, filename)
            dropbox_file_path = os.path.join(dropbox_dir_path, filename).replace('\\', '/')
            file_size = os.path.getsize(local_file_path)

            # Upload file using chunked upload if it exceeds 150 MB, otherwise use standard upload
            if file_size > 150 * 1024 * 1024:  # 150 MB limit for Dropbox API
                print(f"File {local_file_path} exceeds 150 MB, using chunked upload.")
                upload_large_file_to_dropbox(local_file_path, dropbox_file_path)
            else:
                print(f"File {local_file_path} is smaller than 150 MB, using standard upload.")
                with open(local_file_path, "rb") as f:
                    dbx.files_upload(f.read(), dropbox_file_path, mode=dropbox.files.WriteMode.overwrite)

def save_model_to_dropbox(model, processor, output_dir, dropbox_dir):
    """Save model and processor locally, then upload to Dropbox"""
    # Save the model and processor locally first
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    print(f"📁 Model saved locally to {output_dir}")
    print(f"📤 Uploading to Dropbox: {dropbox_dir}")

    # Upload all files in the directory to Dropbox
    upload_directory_to_dropbox(output_dir, dropbox_dir)
    print(f"✅ Model uploaded to Dropbox: {dropbox_dir}")

def upload_logs_to_dropbox(log_file_path, dropbox_log_dir):
    """Upload log file to Dropbox"""
    if os.path.exists(log_file_path):
        dropbox_file_path = os.path.join(dropbox_log_dir, os.path.basename(log_file_path)).replace('\\', '/')
        upload_to_dropbox(log_file_path, dropbox_file_path)
    else:
        print(f"⚠️ Log file not found: {log_file_path}")