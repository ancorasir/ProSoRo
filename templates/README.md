# Templates

## Overview

This directory contains the templates that are used in the project. The structure of the directory is as follows:

```plaintext
templates/
├── README.md
├── cylinder/
│   ├── cylinder.cae                # Abaqus/CAE file
│   ├── template.inp                # Abaqus input file template
│   ├── element.txt                 # Elements
│   ├── node.txt                    # Nodes
│   ├── triangles.txt               # Triangles
│   ├── surf_index.txt              # Surface index
│   ├── surf_node.txt               # Surface nodes
│   └── surf_triangles.txt          # Surface triangles
└── ...
```

## Template Example

Each template is defined in a separate directory named by the module type, and the template files are stored in the directory. The template files include:

- `.cae`: Abaqus/CAE file that defines the simulation and generated the `.inp` file.
- `.inp`: Abaqus input file template that contains the simulation settings.
- `elements.txt`: Definition of all elements, including the element number and corresponding nodes.
- `node.txt`: Definition of all nodes, including the node number and coordinates.
- `triangles.txt`: Definition of all triangles, including the triangle number and corresponding nodes.
- `surf_index.txt`: Definition of the surface nodes' index.
- `surf_node.txt`: Definition of the surface nodes, including the node number and coordinates.
- `surf_triangles.txt`: Definition of the surface triangles, including the triangle number and corresponding nodes.

## Template Generation

Please follow the [guideline](../guideline.ipynb) to generate the templates.
