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
