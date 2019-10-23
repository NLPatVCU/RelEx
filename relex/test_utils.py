
from utils import List_To_File, Output_To_File, Read_From_file
from utils.Punctuation import remove_punctuation, replace_punctuation

print(Read_From_file('../data/P_P/labels_train'))

List_To_File('list.txt', [1, 2, 4])

print(remove_punctuation("Joking!!!!!"))

print(replace_punctuation("Cool!?"))
