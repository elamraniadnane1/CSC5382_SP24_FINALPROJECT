import nbformat
from nbconvert import PythonExporter

def convert_notebook_to_script(notebook_path, script_path):
    # Load the notebook
    with open(notebook_path) as fh:
        nb = nbformat.reads(fh.read(), nbformat.NO_CONVERT)

    # Convert to Python script
    exporter = PythonExporter()
    source, meta = exporter.from_notebook_node(nb)

    # Write the Python script file
    with open(script_path, 'w+') as fh:
        fh.writelines(source)

# Specify the notebook path and desired script path
notebook_path = '/path/to/your/notebook.ipynb'
script_path = '/path/to/your/output_script.py'

# Convert the notebook
convert_notebook_to_script(notebook_path, script_path)
