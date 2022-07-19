import os
import glob

class FileManager:
    def get_images_list(self, path):
        images_list = glob.glob(path)
        return images_list
    
    def change_file_name(self, file_name, before_word, change_word, os_rename=False):
        after_file_name = file_name.replace(before_word, change_word)
        
        if os_rename == True:
            os.rename(file_name, after_file_name)
        return after_file_name
    
    
