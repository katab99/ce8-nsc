import matplotlib.pyplot as plt

def add_new_row(method, dt, t):
    if dt == None:
        dt = "default"
    
    return {
        'method' : [method],
        'dtype' : [dt],
        'time' : [t]
    }

# for visulalizing the results of algorithms
def plotting(res, title, size_params):
    plt.figure(figsize=(8,8))
    plt.title(title)
    plt.imshow(res, cmap='inferno', extent=[size_params[0], size_params[1], size_params[2], size_params[3]])