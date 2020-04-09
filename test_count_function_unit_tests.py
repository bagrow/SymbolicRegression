from count_function_unit_tests import get_phrase_count, get_function_unittest_counts

import os

# new/updated unit test
def test_get_phrase_count():

	# create file an put some number of phrases in it
	file = 'test.txt'
	phrase = 'phrase'

	file_contents = '\n'.join(['This is the file and it contains phrases 6 times',
							   'phrase phrase',
							   'phrase',
							   'phase is a typo',
							   'phrase',
							   '# phrase #'])


	with open(file, 'w') as f: 
		f.write(file_contents)

	output = get_phrase_count(phrase, file)

	# delete the file
	os.remove(file)

	assert output == 6
