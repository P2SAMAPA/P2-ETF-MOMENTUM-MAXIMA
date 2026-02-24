import gitlab
import os
import base64

def sync():
    # 1. Setup Connection using your Secret
    # Ensure this matches the name in your GitHub Secrets
    token = os.getenv('GITLAB_API_TOKEN')
    
    # This path matches your GitLab URL: 
    # https://gitlab.com/p2samapa-group/P2SAMAPA-P2-ETF-MOMENTUM-MAXIMA
    project_path = 'p2samapa-group/P2SAMAPA-P2-ETF-MOMENTUM-MAXIMA'
    
    gl = gitlab.Gitlab('https://gitlab.com', private_token=token)
    
    try:
        project = gl.projects.get(project_path)
        file_path = 'etf_momentum_data.parquet'
        
        # Verify the file exists locally before trying to read it
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found. Did ingestor.py run successfully?")
            return

        # Read the local binary file
        with open(file_path, 'rb') as f:
            file_content = f.read()

        # 2. Check if file already exists to decide Create vs Update
        try:
            # Check if the file is already in the GitLab repo
            existing_file = project.files.get(file_path=file_path, ref='main')
            
            # Update existing file
            existing_file.content = base64.b64encode(file_content).decode('utf-8')
            existing_file.save(
                branch='main',
                commit_message='Update ETF Momentum Data [Automated]',
                encoding='base64'
            )
            print("Successfully UPDATED data in GitLab.")
            
        except gitlab.exceptions.GitlabGetError:
            # Create new file if it doesn't exist in the repo yet
            project.files.create({
                'file_path': file_path,
                'branch': 'main',
                'content': base64.b64encode(file_content).decode('utf-8'),
                'encoding': 'base64',
                'commit_message': 'Initial ETF Momentum Data Upload [Automated]'
            })
            print("Successfully CREATED data in GitLab.")

    except Exception as e:
        print(f"Sync Failed: {e}")

if __name__ == "__main__":
    sync()
