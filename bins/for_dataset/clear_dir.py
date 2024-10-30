import os
import shutil

LANG = "ZH"

dir_path = f"/project/tts/students/yining_ws/multi_lng/F5-TTS/emilia/{LANG}"
# Create a list of filenames from 1 to 89 with leading zeros
expected_files = [f"{LANG}-B{str(i).zfill(6)}" for i in range(0, 100)]
print('expected_files: ', expected_files)

for path in expected_files:
    src_path = os.path.join(dir_path, path)
    # # Delete the JSON file after processing
    # os.rmdir(src_path)
    try:
        shutil.rmtree(src_path)
        print(f"Directory '{src_path}' has been removed successfully")
    except OSError as e:
        print(f"Error: {src_path} : {e.strerror}")
