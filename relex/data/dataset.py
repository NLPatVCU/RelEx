# Inspired from MedaCy

import os
class Dataset:

    def __init__(self, data_directory, text_extension="txt", ann_extension="ann"):
        self.data_directory = data_directory
        self.txt_extension = text_extension
        self.ann_extension = ann_extension
        self.all_data_files = []
        self.index = 0

        all_files_in_directory = os.listdir(self.data_directory)

        # start by filtering all raw_text files, both training and prediction directories will have these
        raw_text_files = sorted([file for file in all_files_in_directory if file.endswith(self.txt_extension)])

        # required ann files for this to be a training directory
        ann_files = [file.replace(".%s" % self.txt_extension, ".%s" % self.ann_extension) for file in raw_text_files]

        # only a training directory if every text file has a corresponding ann_file
        self.is_training_directory = all([os.path.isfile(os.path.join(data_directory, ann_file)) for ann_file in ann_files])

        # set all file attributes except metamap_path as it is optional.
        for file in raw_text_files:
            file_name = file[:-len(self.txt_extension) - 1]
            raw_text_path = os.path.join(data_directory, file)

            if self.is_training_directory:
                annotation_path = os.path.join(data_directory, file.replace(".%s" % self.txt_extension,
                                                                            ".%s" % self.ann_extension))
            else:
                annotation_path = None
            self.all_data_files.append((file_name, raw_text_path, annotation_path))


    def get_data_files(self):
        return self.all_data_files

    def __iter__(self):
        return self

    def __next__(self):
        try:
            word = self.all_data_files[self.index]
        except IndexError:
            raise StopIteration()
        self.index += 1
        return word
