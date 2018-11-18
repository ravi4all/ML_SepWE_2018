#!C:/Python36/python.exe
import cgi
import algorithm

form = cgi.FieldStorage()

review = form.getvalue('review')
prediction = algorithm.test(review)

print("Content-type:text/html\r\n\r\n")
print("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Title</title>
</head>
<body>
<h1>Your review was</h1>
<p>{}</p>
<p>Your review is : <b>{}</b></p>
</body>
</html>
""".format(review, prediction))