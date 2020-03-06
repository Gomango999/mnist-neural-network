from tkinter import *
from PIL import ImageDraw, Image
from util import *
import numpy as np

root = Tk()
root.title("MNIST NN")

canvas_width = 28*10
canvas_height = 28*10
canvas = Canvas(root, width=canvas_width, height=canvas_height, bg='white')
canvas.pack()

class g():
	points = []
	user_line = None
	drawing = False
	number_image = Image.new("RGB", (canvas_width, canvas_height), "white")
	draw = ImageDraw.Draw(number_image)
v = StringVar()

def analyse():
	small_width = 28
	small_height = 28
	small_image = g.number_image.resize((small_width,small_height), Image.ANTIALIAS)

	np_image = np.array(small_image)
	np_image = np_image[:, :, 0]
	np_image = (255 - np_image) / 255.0
	np_image = np_image.ravel().reshape(small_width*small_height,1)
	
	neural_network = load("mnist.nn")
	output = neural_network.run(np_image)
	v.set("My guess: " + str(classify(output)))

	canvas.delete("all")
	g.draw.rectangle([0,0,canvas_width,canvas_height], fill="white")


button = Button(root, text="Analyse", command=analyse)
button.pack()

output_label = Label(root, textvariable=v)
output_label.pack()

def leftClick(event):
	g.points = []
	g.user_line = None
	g.drawing = True
	g.points.append(event.x)
	g.points.append(event.y)

# It seems that the append is causing slow downs...
# Perhaps restart it when it is a certain length, because it will never be a word?
def leftMove(event):
	if g.drawing: 
		g.points.append(event.x)
		g.points.append(event.y)
		if g.user_line == None:
			g.user_line = canvas.create_line(g.points, width=20)
		else:
			canvas.coords(g.user_line, g.points)
			g.draw.line(g.points[-4:], "black", width=20)
	
def leftRelease(event):
	g.points = []
	g.user_line = None

canvas.bind('<Button-1>', leftClick)
canvas.bind('<B1-Motion>', leftMove)
canvas.bind('<ButtonRelease-1>', leftRelease)

root.mainloop()