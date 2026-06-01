import unittest
from unittest.mock import patch, mock_open
from seeker.util import purge

class TestPurgeFunction(unittest.TestCase):

    @patch('seeker.util.listdir')
    @patch('builtins.open', new_callable=mock_open)
    def test_purge_with_non_utf8_file(self, mock_file, mock_listdir):
        mock_listdir.return_value = ['good_snippet.txt', 'bad_snippet.txt']
        mock_file.side_effect = [
            mock_open(read_data='Good snippet content').return_value,
            IOError('invalid start byte')]  # Simulate non-UTF-8 file

        # Call the function
        purge(['good_snippet.txt', 'bad_snippet.txt'])

        # Check if both files were attempted to be processed
        self.assertEqual(mock_file.call_count, 2)

if __name__ == '__main__':
    unittest.main()
