def autoassign(locs):
    """Assign locals to self."""
    for key in locs.keys():
        if key!="self":
            locs["self"].__dict__[key]=locs[key]
