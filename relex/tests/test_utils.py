import unittest
from unittest import TestCase
# add import statments here


class TestUtils(unittest.TestCase):

    def test_remove_punctuation(self):
        self.assertEqual(remove_punctuation("Hello, world!:)"), "Hello world")

    def test_replace_punctuation(self):
        self.assertEqual(replace_punctuation("Hello, world!:)"), "Hello, world,,,")

    def test_list_to_file(self):
        file = "test_list_to_file.txt"
        list_to_file(file, [1,2,3])

        self.AssertTrue(path.exists(file)) #tests if file is created

        file_opened=open(file, "r")
        file_contents = file_opened.read()
        file_opened.close()
        self.assertEqual(file_contents, "1\n2\n3\n") #tests if file contents is correct

    def test_read_from_file(self):
        self.assertEqual(read_from_file("read_from_file_test.txt"), "Hello World")

    # test desample
    # test gcd
    # test unique_labels
    # test output_to_file 

if __name__ == '__main__':
    unittest.main()
