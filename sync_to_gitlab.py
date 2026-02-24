import gitlab
import os
import base64

def sync():
    # 1. Setup Connection using Secret
    token = os.getenv('GITLAB_API_TOKEN')
    project_path = 'p2samapa-group/P2SAMAPA-P2-ETF-MOMENTUM-MAXIMA'
    
    # Initialize GitLab with a timeout to prevent workflow hangs
    gl = gitlab.Gitlab('https://gitlab.com', private_token=token, timeout=30)
    
    try:
        project = gl.projects.get(project_path)
        file_path = 'etf_momentum_data.parquet'
        
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found. Ingestor likely failed.")
            return

        # Read the local binary file
        with open(file_path, 'rb') as f:
            file_content = f.read()

        # Encode content to base64 for GitLab API
        base64_content = base64.b64encode(file_content).decode('utf-8')

        # 2. Check if file exists to decide Create vs Update
        try:
            existing_file = project.files.get(file_path=file_path, ref='main')
            
            # Update existing file with safety check
            existing_file.content = base64_content
            existing_file.save(
                branch='main',
                commit_message='Update ETF Momentum Data [Automated Sync]',
                encoding='base64'
            )
            print("✅ Successfully UPDATED data in GitLab.")
            
        except gitlab.exceptions.GitlabGetError:
            # Create new file if it doesn't exist
            project.files.create({
                'file_path': file_path,
                'branch': 'main',
                'content': base64_content,
                'encoding': 'base64',
                'commit_message': 'Initial ETF Momentum Data Upload [Automated Sync]'
            })
            print("✅ Successfully CREATED data in GitLab.")

    except Exception as e:
        print(f"❌ Sync Failed: {e}")

if __name__ == "__main__":
    sync()
