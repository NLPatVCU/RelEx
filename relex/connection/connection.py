from segment import Segmentation

def read_from_file(file):
    """
    Reads external files and insert the content to a list. It also removes whitespace
    characters like `\n` at the end of each line

    :param file: name of the input file.
    :return : content of the file in list format
    """
    if not os.path.isfile(file):
        raise FileNotFoundError("Not a valid file path")

    with open(file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    return content

class Connection:
    def __init__(self, CSV=False, segment=False, dataset=None, sentences=None, labels=None, preceding_segs=None, middle_segs=None, succeeding_segs=None, concept1_segs=None, concept2_segs=None):
        self.CSV = CSV
        self.segment = segment
        if self.CSV:
            self.sentences = sentences
            self.labels = labels
            if self.segment:
                self.preceding_segs = preceding_segs
                self.middle_segs = middle_segs
                self.succeeding_segs = succeeding_segs
                self.concept1_segs = concept1_segs
                self.concept2_segs = concept2_segs
        else:
            self.dataset = dataset

        def get_data_object(self):

            if self.CSV:
                object = {'preceding': [], 'concept1': [], 'concept2': [], 'middle': [], 'succeeding': [], 'sentence': [],'label': []}
                train_data = read_from_file(self.sentences)
                train_labels = read_from_file(self.labels)
                object['sentence'].append(self.sentences)
                object['label'].append(self.labels)

                if self.segment:
                    train_preceding = read_from_file(self.preceding_segs)
                    train_middle = read_from_file(self.middle_segs)
                    train_succeeding = read_from_file(self.succeeding_segs)
                    train_concept1 = read_from_file(self.concept1_segs)
                    train_concept2 = read_from_file(self.concept2_segs)

                    object['preceding'].append(self.preceding_segs)
                    object['concept1'].append(self.concept1_segs)
                    object['middle'].append(self.middle_segs)
                    object['concept2'].append(self.concept2_segs)
                    object['succeeding'].append(self.succeeding_segs)

                return object

            else:
                object = Segmentation(self.dataset)
                
