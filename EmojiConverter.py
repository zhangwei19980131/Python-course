message = input(">")
words = message.split(' ')
emojis = {
    ":)": "☻",
    ":(": "o(╥﹏╥)o"
}
output = ""
for word in words:
    output += emojis.get(word, word) + " "
print(output)
