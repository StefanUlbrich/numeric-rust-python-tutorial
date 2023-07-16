# Tutorial for writing numerical Python extensions with Rust and `ndarray`

## Initialize the Rust project

Set up an empty repository 

```sh
mkdir mkdir rust-python-ndarray-tutorial 
cd mkdir rust-python-ndarray-tutorial
cargo init --lib --vcs git
```

And add a few that we will use at the beginning of the tutorial


```sh
cargo add ndarray -F rayon # numerical crate
cargo add --dev tracing # for debug messages
cargo add itertools # better handling of iterators
```

Let's add a few changes to make python and rust coexist

```sh
git mv src/lib.rs src/rust
```

and add the following section to the `cargo.toml`

```toml
[lib]
path = "src/rust/lib.rs" 
```

## Initialize Python project

We'll leverage [Poetry](https://python-poetry.org/) for dependency handling
and virtual environment management

```sh
poetry init .
poetry install
```

Finally, let's add a `.gitignore` file from 
[here](https://www.toptal.com/developers/gitignore/api/rust,python,macos,linux,jupyternotebooks).
