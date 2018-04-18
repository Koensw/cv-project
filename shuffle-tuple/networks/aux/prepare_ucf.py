import re
import os

input_dir = "../../../ucf101"
output_dir = "./data"

with open("train01_image_keys.txt") as fl:
    line = fl.readline()
    cnt = 1
    while line:
        name = line.split()[0];
        result = re.match("^v_([a-zA-Z]+)_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)/images/([a-zA-Z0-9_.]+)_[0-9_]+", name)
        
        folder, name1, name2, avi = result.groups()
        full_dir = input_dir + '/' + folder
        
        out_dir = output_dir + '/' + "v_{}_{}_{}/images".format(folder, name1, name2)
        command = "ffmpeg -i " + (full_dir + '/' + avi) + " -q:a 1 -f image2 " + out_dir + '/' + avi + "_%06d.jpg"
        if not os.path.exists(out_dir):
            print("[{}/{}] Creating images for".format(cnt, 994250), avi)
            os.makedirs(out_dir)
            os.system(command + " > /dev/null 2>&1")
            line = fl.readline()
        
        if os.listdir(out_dir) == []:
            print("[{}/{}] ERROR: no images for".format(cnt, 994250), avi, "from", command, "retrying...")
            os.rmdir(out_dir)
        else:
            line = fl.readline()
            cnt = cnt + 1
    
