
class Utils:
    def __init__(self):
        pass
    def ann_to_json(self, annotation_text):
        """
        Takes annotation text and converts it to json

        :param annotation_text: annotation file text
        :return annotation object (in json format)
        """
        annotations = {'entities': {}, 'relations': []}
        for line in annotation_text.split("\n"):
            if "\t" in line:
                line = line.split("\t")

                # To obtain the entities
                if 'T' in line[0]:
                    entity_id = line[0]
                    entity_span = line[-1]

                    tags = line[1].split(" ")
                    entity_name = tags[0]
                    entity_start = int(tags[1])
                    entity_end = int(tags[-1])
                    annotations['entities'][line[0]] = (entity_id, entity_start, entity_end, entity_name, entity_span)

                # To obtain the relations
                if 'R' in line[0]:
                    relation_id = line[0]
                    tags = line[1].split(" ")
                    relation_name = tags[0]
                    relation_start = tags[1].split(':')[1]
                    relation_end = tags[2].split(':')[1]
                    annotations['relations'].append((relation_id, relation_name, relation_start, relation_end))
        return annotation

    def read_from_file(self, file):
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

    def output_to_file(self, true, pred, target):
        """
        Function to create .txt file and csv file of classification report
        """
        report = classification_report(true, pred, target_names=target)
        report_dict = classification_report(true, pred, target_names=target, output_dict=True)
        df_report = pd.DataFrame(report_dict).transpose()

        #writes .txt file with results
        txt_file = open(output_txt_path, 'a')
        txt_file.write(report)
        txt_file.close()

        # writes csv file
        csv_report = df_report.to_csv()
        csv_file = open(output_csv_path, 'a')
        csv_file.write(csv_report)
        csv_file.close()


    def list_to_file(self, file, input_list):
        """
        Function to write the contents of a list to a file

        :param file: name of the output file.
        :param input_list: list needs to be written to file
        """
        with open(file, 'w') as f:
            for item in input_list:
                f.write("%s\n" % item)

    def remove_Punctuation(self, string):
        """
        Function to remove punctuation from a given string. It traverses the given string
        and replaces the punctuation marks with null

        :param string: given string.
        :return string:string without punctuation
        """
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

        for x in string.lower():
            if x in punctuations:
                string = string.replace(x, "")

        return string

    def replace_Punctuation(self, string):
        """
        Function to remove punctuation from a given string. It traverses the given string
        and replaces the punctuation marks with comma (,)

        :param string: given string.
        :return string:string without punctuation
        """
        punctuations = '''!()-[]{};:'"\,<>/?@#$%^&*_~'''

        for x in string.lower():
            if x in punctuations:
                string = string.replace(x, ",")

        return string
