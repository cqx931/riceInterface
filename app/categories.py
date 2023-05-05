categories = [
    {
        "index": 0,
        "name": "branches",
        "detections": [],
        "f": lambda s: s.count('|') == 1 and s.count('+') >= 1, # branches
    },
    {
        "index": 1,
        "name": "The Delicate Heart",
        "detections": [],
        "f": lambda s: s == "*" or s == "*º" or s == "º*", # delicate heart
    },
    {
        "index": 2,
        "name": "Hidden Balance",
        "detections": [],
        "f": lambda s: s.count('*') == 2 and s.count('_') == 0 and s.count('|') == 0, # hiddem balance
    },
    {
        "index": 3,
        "name": "The Mist",
        "detections": [],
        "f": lambda s: s == "", # The Mist - empty
    },
    {
        "index": 4,
        "name": "The Broken Cloud exception",
        "detections": [],
        "f": lambda s: s.count('_') > 5, # The Broken Cloud
    },
    {
        "index": 5,
        "name": "The Libra",
        "detections": [],
        "f": lambda s: s.count('*') == 2 and s.count('_') == 1, # libra
    },
    {
        "index": 6,
        "name": "The Divider",
        "detections": [],
        "f": lambda s: s.count('_') == 1 and s.count('|') == 0, # divider
    },
    {
        "index": 7,
        "name": "Mirror",
        "detections": [],
        "f": lambda s: s.count('_') == 2 and s.count('|') == 0, # the mirror
    },
    {
        "index": 8,
        "name": "The Three Parallels",
        "detections": [],
        "f": lambda s: s.count('_') == 3 and s.count('|') == 0, # The Three Parallels
    },
    {
        "index": 9,
        "name": "Four cracks",
        "detections": [],
        "f": lambda s: s.count('_') == 4 and s.count('|') == 0, # Four cracks
    },
    {
        "index": 10,
        "name": "The Clustered Five",
        "detections": [],
        "f": lambda s: s.count('_') == 5, # The Clustered Five
    },
    {
        "index": 11,
        "name": "Curvy lines",
        "detections": [],
        "f": lambda s: s.count('+') >= 3 and s.count('|') >= 1, # branches
    },
]

def detect_category(s):
    for category in categories:
        if category["f"](s):
            return category["name"]
    return "The Mist"  # If the string does not belong to any category