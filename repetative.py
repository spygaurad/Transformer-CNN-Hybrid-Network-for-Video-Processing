def find_repetitive_words(text, l, s):
    w = l+s
    sliding_window = []
    sliding_window[l:] = text[:l]
    print(sliding_window)



text = "repetitive repeat"
output = find_repetitive_words(text, 4, 6)
print(output)