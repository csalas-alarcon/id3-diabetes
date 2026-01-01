
# Entropy Calculation ->

entropy = sum([
    -count / len(x_ids) * math.log(count / len(x_ids), 2)
    if count else 0
    for count in label_count
])
