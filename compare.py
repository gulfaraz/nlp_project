import pickle

(a1, b1, c1, d1, e1, f1) = pickle.load(open("gru regression/created_dics.pickle","rb"))
(a2, b2, c2, d2, e2, f2) = pickle.load(open("nlp preprocessing caption/created_dics.pickle","rb"))

true_true_count = 0
true_false_count = 0
false_true_count = 0
false_false_count = 0
equal_count = 0
unequal_count = 0
regression_count = 0
classification_count = 0
for index, value in enumerate(f1):
	if f1[index] == True:
		regression_count += 1
	if f2[index] == True:
		classification_count += 1
	if f1[index] == True and f2[index] == True:
		true_true_count += 1
		equal_count += 1
	if f1[index] == True and f2[index] == False:
		true_false_count += 1
		unequal_count += 1
	if f1[index] == False and f2[index] == True:
		false_true_count += 1
		unequal_count += 1
	if f1[index] == False and f2[index] == False:
		false_false_count += 1
		equal_count += 1

print(true_true_count)
print(true_true_count/len(f1))
print(true_false_count)
print(true_false_count/len(f1))
print(false_true_count)
print(false_true_count/len(f1))
print(false_false_count)
print(false_false_count/len(f1))
print(equal_count)
print(equal_count/len(f1))
print(unequal_count)
print(unequal_count/len(f1))
print(regression_count)
print(classification_count)
