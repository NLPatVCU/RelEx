
from utils import read_from_file, list_to_file, remove_punctuation, replace_punctuation, desample, desample_given_unique_labels_current_ratios
# print("testing read from file...")
# content = read_from_file('../data/P_P/sentence_train')
# print(content)

# print("testing punctuation...")
# print(replace_punctuation("!!!???hjjf??nhg?"))
# print(remove_punctuation("?jhgy"))

# print("testing list to file")
# print(list_to_file("test.txt", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
labels = ["AA", "BB", "AA", "BB", "BB", "BB"]
sentences = ["sen1", "sen2", "sen3", "sen4", "sen5"]

print("Testing desample...")
print(desample_given_unique_labels_current_ratios(sentences, labels, [1,1], ["AA", "BB"], [2,3]))
