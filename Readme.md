# <span style="color:red;"> The Rust and the Python </span>

*A tutorial for writing numerical Python extensions with Rust and `ndarray`*


This session is a special technical session that features a tutorial and
introduction on coding machine learning algorithms and other computationally
extensive software. Python is the de-facto standard in Machine Learning and Data
Science, and Rust is a very modern, and fast systems programming language that
has been elected the most beloved programming the last 5 years in a row on
stackoverflow. " .. but written in Rust" became a common news headline. Let's
explore how both languages can benefit from each other!

The session covers the evolution of a (simple but complete) machine learning
algorithm from scratch. We start with the translation of formulas into clean,
modern and efficient code, modern packaging ready for upload to a package
registry, and the successive translation of parts of the algorithms to a
compiled language for performance gains. A performance analysis will reveal
whether the Python's reputation for being slow is justified and may hold some
surprise. There are many tutorials out there for connecting Rust with Python but
only few focus on numeric computation and numpy compatibility like this one. All
code is published on GitHub and may be used as a template for your own projects.

![AI-generated image about a dark cult of the crab](tutorial/cult_sm.png)

## Setup the Python package


```sh
$ # Python packaging and dependency management
$ curl -sSL https://install.python-poetry.org | python3 -
$ # Get the code
$ git clone git@github.com:StefanUlbrich/numeric-rust-python-tutorial.git tutorial
$ cd tutorial && git checkout python-skeleton
tutorial$ # Create virtual environment and install dependencies
tutorial$ poetry env use python3.11 # Optional
tutorial$ poetry install
```

## Setup the Rust benchmark

```sh
$ # Installation
$ curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
$ rustup update # Optional: Update the tool chain
$ cd tutorial && git checkout rust-examples
tutorial$ # git checkout rust-implementation # spoiler alert!
tutorial$ (cd data; poetry run data) # we need data for the experiments
tutorial$ cargo run --example read_data
tutorial$ # cargo bench # run benchmarks later
```

## Setup the extension

```sh
tutorial$ git checkout extension-skeleton
tutorial$ # git checkout extension-final # spoiler alert!
tutorial$ maturin develop -r --strip # Builds the extensions and adds it to the venv
tutorial$ maturin build -r --strip # Creates a binary wheel
```