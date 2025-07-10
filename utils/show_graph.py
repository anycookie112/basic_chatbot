from PIL import Image
from io import BytesIO


def show_mermaid(react_graph):

    png_bytes = react_graph.get_graph(xray=True).draw_mermaid_png()

    # Load into a PIL image
    image = Image.open(BytesIO(png_bytes))

    # Display using PIL (launches the default image viewer)
    image.show()



def show_supervisor(supervisor):
    """
    Display the LangGraph supervisor graph using PIL (for VS Code or script environments).
    
    Args:
        supervisor: The LangGraph supervisor object with .get_graph().draw_mermaid_png().
    """
    png_data = supervisor.get_graph().draw_mermaid_png()
    image = Image.open(BytesIO(png_data))
    image.show()
