# My implimentation of the QuadTree data structure
import networkx as nx

class Particle:
    def __init__(self):
        self.x = None
        self.y = None 
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def update(self, x, y):
        self.x = x
        self.y = y
    def isBound(self, bounds):
        if self.x >= bounds[0] and self.x < bounds[1] and self.y >= bounds[2] and self.y < bounds[3]:
            return True
        return False

class Node:
    def __init__(self) -> None:
        self.isLeaf = False 
        self.children = None
        self.particles = None
        self.bounds = None

    def buildNode(self, particles, bounds, maxParticles=1):
        self.bounds = bounds
        if len(particles) == 0:
            self.isLeaf = True
            return 
        if len(particles) <= maxParticles:
            self.isLeaf = True
            self.particles = [particle for particle in particles]
            return
        self.isLeaf = False
        # Generate 4 children nodes
        self.children = [Node(), Node(), Node(), Node()]
        delX = (bounds[1] - bounds[0])/2
        delY = (bounds[3] - bounds[2])/2
        n = 0
        for i in range(1, -1, -1):
            for j in range(2):
                childBounds = [bounds[0] + j*delX, bounds[0] + (j+1)*delX, bounds[2] + i*delY, bounds[2] + (i+1)*delY]
                childParticles = [particle for particle in particles if particle.isBound(childBounds)]
                self.children[n].buildNode(childParticles, childBounds, maxParticles)
                n += 1






class QuadTree:
    def __init__(self):
        self.root = None
        self.maxDepth = None
        self.maxParticles = None
        self.particles = None
        self.xMin = None
        self.xMax = None
        self.yMin = None
        self.yMax = None
    def buildTree(self):
        self.root = Node()
        self.root.buildNode(self.particles, [self.xMin, self.xMax, self.yMin, self.yMax], self.maxParticles)


# Test code

# Generate random particles
import random
import matplotlib.pyplot as plt
import time

def generateParticles(n, xMin, xMax, yMin, yMax, seed = None):
    if seed != None:
        random.seed(seed)
    particles = []
    for i in range(n):
        particles.append(Particle(random.uniform(xMin, xMax), random.uniform(yMin, yMax)))  
    return particles



import seaborn as sns

def plotParticles(particles, bounds, maxParticles=1):
    # Extract x and y coordinates of particles
    x = [particle.x for particle in particles]
    y = [particle.y for particle in particles]

    # Store the smallest quadrant size
    smallest_quadrant_size = [float('inf')]

    # Setup the plot
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 10))

    def _subdivide_quadrant(ax, qxMin, qxMax, qyMin, qyMax):
        """Recursively subdivide a quadrant."""
        # Calculate the midpoints
        midX = (qxMax + qxMin) / 2
        midY = (qyMax + qyMin) / 2

        # Update the size for each particle in this quadrant
        quadrant_size = max(qxMax - qxMin, qyMax - qyMin)
        smallest_quadrant_size[0] = min(smallest_quadrant_size[0], quadrant_size)

        # Count particles in the quadrant
        count = sum(qxMin <= px < qxMax and qyMin <= py < qyMax for px, py in zip(x, y))
        
        # If 1 or no particles in the quadrant, no further subdivision needed
        if count <= maxParticles:
            return

        # Draw the midlines for the quadrant
        ax.plot([midX, midX], [qyMin, qyMax], color='gray', linestyle='--', alpha=0.5)
        ax.plot([qxMin, qxMax], [midY, midY], color='gray', linestyle='--', alpha=0.5)

        # Recursively subdivide the new quadrants
        _subdivide_quadrant(ax, qxMin, midX, qyMin, midY)
        _subdivide_quadrant(ax, midX, qxMax, qyMin, midY)
        _subdivide_quadrant(ax, qxMin, midX, midY, qyMax)
        _subdivide_quadrant(ax, midX, qxMax, midY, qyMax)

    # Invoke the recursive function with the initial plot boundaries
    xMin, xMax, yMin, yMax = bounds
    _subdivide_quadrant(ax, xMin, xMax, yMin, yMax)

    # Calculate marker size ensuring non-overlap
    marker_size = 50 * (smallest_quadrant_size[0])

    # Plot the particles using a Seaborn palette
    ax.scatter(x, y, color=sns.color_palette("pastel")[0], s=marker_size, edgecolor="black", linewidth=0.5)

    ax.set_xlim(xMin, xMax)
    ax.set_ylim(yMin, yMax)
    ax.set_title("Particles Visualization with Quadtree")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    
    # Reduce the number of ticks for clarity
    ax.set_xticks([xMin, xMax])
    ax.set_yticks([yMin, yMax])

    plt.tight_layout()
    plt.show()







# Generate 10 particles and visualize 

xMin = -10
xMax = 10
yMin = -10
yMax = 10
seed = 42069

particles = generateParticles(10, xMin, xMax, yMin, yMax, seed)

# Now build the tree
quadTree = QuadTree()
quadTree.particles = particles
quadTree.xMin = xMin
quadTree.xMax = xMax
quadTree.yMin = yMin
quadTree.yMax = yMax
quadTree.maxParticles = 2
quadTree.buildTree()

plotParticles(particles, [xMin, xMax, yMin, yMax], quadTree.maxParticles)





import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

G = nx.Graph()
node_counter = [0]
node_colors = []
node_alphas = []
node_depths = {}
nodes_with_particles = []
edge_alphas = {}

# Fetch a set of colors from Seaborn's color palette
depth_colors = sns.color_palette("husl", 10)

def plotTree(node, parent=None, depth=0):
    node_id = node_counter[0]
    node_counter[0] += 1
    G.add_node(node_id)
    
    color = depth_colors[depth % len(depth_colors)]
    node_colors.append(color)
    node_depths[node_id] = depth
    
    if node.particles:
        nodes_with_particles.append(node_id)
        node_alphas.append(1)
    elif node.isLeaf:
        node_alphas.append(0.2)
    else:
        node_alphas.append(1)
    
    if parent is not None:
        G.add_edge(node_id, parent)
        if node.isLeaf and not node.particles:
            edge_alphas[(node_id, parent)] = 0.2
        else:
            edge_alphas[(node_id, parent)] = 1
    if not node.isLeaf:
        for child in node.children:
            plotTree(child, node_id, depth=depth+1)

from networkx.drawing.nx_pydot import graphviz_layout

plotTree(quadTree.root)
pos = graphviz_layout(G, prog="dot")

# Setting the figure size
plt.figure(figsize=(15, 10))

# Drawing edges with assigned alphas
nx.draw_networkx_edges(G, pos, width=1.5, edge_color='gray', alpha=0.2)
for edge, alpha in edge_alphas.items():
    nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color='gray', alpha=alpha)

# Drawing nodes with adjusted size
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.2)
for node_id, alpha in enumerate(node_alphas):
    nx.draw_networkx_nodes(G, pos, nodelist=[node_id], node_color=node_colors[node_id], node_size=300, alpha=alpha)

# Highlight nodes with particles
nx.draw_networkx_nodes(G, pos, nodelist=nodes_with_particles, node_color="black", node_size=25)

plt.axis('off')
plt.tight_layout()
plt.show()
