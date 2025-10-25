REPO_ID = "adyen/DABstep"

# Restricted imports for non-sandboxed environments (running directly on host)
RESTRICTED_AUTHORIZED_IMPORTS = ["numpy", "pandas", "json", "csv", "glob", "markdown", "os", "io", "pathlib"]

# Comprehensive imports allowed when running in a sandboxed container
CONTAINER_AUTHORIZED_IMPORTS = [
    "os", "sys", "pathlib", "glob", "shutil", "tempfile",
    "json", "csv", "pickle", "yaml", "toml",
    "numpy", "pandas", "scipy", "sklearn",
    "matplotlib", "seaborn", "plotly",
    "requests", "urllib", "http", "socket",
    "datetime", "time", "calendar",
    "re", "string", "textwrap",
    "io", "logging", "traceback",
    "collections", "itertools", "functools",
    "math", "statistics", "random",
    "subprocess", "threading", "multiprocessing",
    "typing", "dataclasses", "enum",
    "markdown", "html", "xml",
]
