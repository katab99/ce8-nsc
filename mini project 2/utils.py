def add_new_row(method, dt, t):
    if dt == None:
        dt = "default"
    
    return {
        'method' : [method],
        'dtype' : [dt],
        'time' : [t]
    }