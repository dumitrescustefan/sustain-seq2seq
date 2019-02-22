import requests, os, sys

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

# TAKE ID FROM SHAREABLE LINK
file_id = "0BwmD_VLjROrfTHk4NFg2SndKcjQ"
# DESTINATION FILE ON YOUR DISK
destination = "cnn_stories.tgz"
#download_file_from_google_drive(file_id, destination)



file_id = "0BwmD_VLjROrfM1BxdkxVaTY2bWs"
# DESTINATION FILE ON YOUR DISK
destination = "dailymail_stories.tgz"
download_file_from_google_drive(file_id, destination)



#echo "Getting cnn-stories (tgz)"
#wget -v https://drive.google.com/open?id=0BwmD_VLjROrfTHk4NFg2SndKcjQ
#echo "Getting dailymailcnn-stories (tgz)"
#wget -v https://drive.google.com/uc?export=download&confirm=8h-c&id=0BwmD_VLjROrfM1BxdkxVaTY2bWs
