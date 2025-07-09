from PIL import Image
from io import BytesIO


def show_mermaid(react_graph):

    png_bytes = react_graph.get_graph(xray=True).draw_mermaid_png()

    # Load into a PIL image
    image = Image.open(BytesIO(png_bytes))

    # Display using PIL (launches the default image viewer)
    image.show()

