import csv, re
from model import Model

spam_model = Model()

with open('spam.csv', encoding='mac-roman') as f:
	reader = csv.reader(f)
	next(reader) #discard header row
	for row in reader:
		label = row[0]
		msg = re.sub(r'[^a-z ]+', '', row[1].lower())
		spam_model.observe((label, {'msg': msg.split()}))

spam_model.process()
spam_model.train()
print(spam_model.test())

#interactive demo
while True:
	msg = input('Input message to be classified: ')
	msg = re.sub(r'[^a-z ]+', '', msg)
	print(spam_model.predict({'msg': msg.split()}))