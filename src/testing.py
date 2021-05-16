sampled = []
for x in normal_domains:
    word = []
    for y in x:
        word.append(__np_sample(y))
    sampled.append(word)

print("results")
readablen = __to_readable_domain(np.array(sampled), inv_map=data_dict['inv_map'])
