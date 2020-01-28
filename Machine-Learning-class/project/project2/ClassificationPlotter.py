import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplc

def plot_regions(model, X, y, 
                 num_ticks=100, cmap='rainbow', fig_size = None, legend=True, 
                 close=True, display=True, path=None, keras=False):

    # Convert X to numpy array
    X = np.array(X)
    y = np.array(y)
    
    # Check to see if there are exactly 2 features
    if X.shape[1] != 2:
        raise Exception('Training set must contain exactly ' + 
                        'two features.')
        
     
    # Find min and max points for grid axes
    minx, maxx = min(X[:,0]), max(X[:,0])
    marginx = (maxx - minx) / 20
    x0, x1 = minx - marginx, maxx + marginx
    
    miny, maxy = min(X[:,1]), max(X[:,1])
    marginy = (maxy - miny) / 20
    y0, y1 = miny - marginy, maxy + marginy
    
    # Create grid tick marks
    xticks = np.linspace(x0, x1, num_ticks)
    yticks = np.linspace(y0, y1, num_ticks)
    
    # Create array of grid points
    # tile stacks copies of xticks end creating a size num_ticks^2 array
    # repeat repeats each elt of yticks to create a size num_ticks^2 array
    # They are combined into an array of shape (2, num_ticks^2)
    # transpose creates array of pts with shape (num_ticks^2, 2).
    grid_pts = np.transpose([np.tile(xticks,len(yticks)),
                             np.repeat(yticks,len(xticks))])
    
    # Feed grid points to model to generate 1D array of classes
    if(keras==True): 
        class_pts = model.predict_classes(grid_pts)
        class_pts = class_pts.reshape((len(class_pts),))
    else:
        class_pts = model.predict(grid_pts)

    # Get list of classes. This could, in theory, contain text labels.
    classes = np.unique(y)
    k = len(classes)    
        
    # create new list with numerical classes
    class_pts_2 = np.zeros(class_pts.shape)
    for i in range(len(classes)):
        sel = class_pts == classes[i]
        class_pts_2[sel] = i

    # reshape classification array into 2D array corresponding to grid
    class_grid = class_pts_2.reshape(len(xticks),len(yticks) )
    
    # Set a color map        
    my_cmap = plt.get_cmap(cmap)
          
    # Close any open figures and set plot size.
    
    if(close): 
        plt.close()
    if(not fig_size is None):
        plt.figure(figsize=fig_size)
    
    # Add color mesh
    plt.pcolormesh(xticks, yticks, class_grid, cmap = my_cmap, zorder = 1, 
                   vmin=0, vmax=k-1 )
    
    # Add transparency layer to lighten the colors
    plt.fill([x0,x0,x1,x1], [y0,y1,y1,y0], 'white', alpha=0.5, zorder = 2)
    
    # Select discrete cuts for color map
    cuts = np.arange(k) / (k - 1)

    # Add scatter plot for each class, with seperate colors and labels
    for i in range(k):
        sel = y == classes[i]       
        
        my_c = mplc.rgb2hex(my_cmap(cuts[i]))
        plt.scatter(X[sel,0],X[sel,1], c=my_c, edgecolor='k', 
                    zorder=3, label=classes[i])

    if(legend):
        plt.legend()
    
    if(not path is None):
        plt.savefig(path, format='png')

    
    if(display): 
        plt.show()
    