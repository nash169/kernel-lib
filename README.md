# Kernel library
Library containing (Eigen-based) vectorized implementation of some kernel.

#### ToDo list
Here is a list of features to implement or unsolved issues:
- variadic template to extend Expansion to handle reference points dependent kernels;
- decide if memorizing kernel evaluation within the kernel class;
- Add Hessian and Params Hessian in all the kernels
- Add Polynomial and Cosine Kernels
- Force inline function to avoid code repetition
- Template headers separation (https://doc.magnum.graphics/magnum/compilation-speedup.html)

### Authors/Maintainers
- Bernardo Fichera (bernardo.fichera@epfl.ch)

### Available Kernels

#### Invariant-Stationary
- Radial Basis Function (RBF)

### Dependencies
This library depends on Eigen for the linear algebra. The latest git version is required.
```sh
git clone https://gitlab.com/libeigen/eigen.git (git@gitlab.com:libeigen/eigen.git)
cd eigen && mkdir build && cmake .. && (sudo) make install
```

Other (optional) dependencies for improving performances required by Eigen are:
- Linear Algebra -> LAPACK, BLAS, OPENBLAS, ATLAS, MKL
- Multi-threading -> OPENMP (TBB)

### Installation
Clone the repository including the sub-modules
```sh
git clone --recursive https://github.com/nash169/kernel-lib.git (git@github.com:nash169/kernel-lib.git)
```
**kernel-lib** relies on WAF compilation tool.
Arch provides an updated version of WAF exec in the standard repo
```sh
sudo pacman -S waf
```
For other distros it is better to download the latest version from the official website and move the executable in the library repo
```sh
wget 'https://waf.io/waf-2.0.21'
mv waf-2.0.19waf-2.0.19 waf && mv waf /path/to/kernel-lib
cd /path/to/kernel-lib
chmod +x waf
```
Compile and install using waf commands
```sh
waf (./waf) configure build
```
or
```sh
waf (./waf) configure && waf (./waf)
```
Install the library (optional)
```sh
(sudo) waf (./waf) install
```
If you want to make clean installation
```sh
(sudo) waf (./waf) distclean configure build install
```

#### Compilation options
In order to set the desired compiler define the environment variable CXX=<g++,clang++,icpc> (gnu, clang and intel compiler respectively).

It is highly recommended to compile with AVX support
```sh
waf (./waf) configure --release
```
Activate multi-threading outside EIGEN
```sh
waf (./waf) configure --multi-threading
```
Compile static library (default option)
```sh
waf (./waf) configure --static
```
Compile shared library
```sh
waf (./waf) configure --shared
```
##### EIGEN derived options
Enable OPENMP multi-threading
```sh
waf (./waf) configure --eigen-openmp
```
Enable LAPACK
```sh
waf (./waf) configure --eigen-lapack
```
Enable BLAS
```sh
waf (./waf) configure --eigen-blas=<blas|openblas|atlas>
```
Enable MKL
```sh
waf (./waf) configure --eigen-mkl
```
##### MKL derived options
By default MKL uses `sequential` option. If you choose OpenMP multi-threading it is possible select between the GNU (default), `gnu`, or Intel, `intel`, version through `--mkl-openmp` option.
```sh
waf (./waf) configure --mkl-threading=<sequential|openmp|tbb> -mkl-openmp=<gnu|intel> --mkl-64=<true|false>
```
##### Suggested configurations
```sh
waf (./waf) configure --release --multi-threading --eigen-openmp --eigen-lapack --eigen-blas=blas
```
```sh
waf (./waf) configure --release --multi-threading --eigen-openmp --eigen-mkl --mkl-threading=tbb --mkl-64
```

### Examples
```sh
./build/src/examples/create_kernels
```

### Benchmarks
```sh
./build/src/benchmarks/bench_rbf
```

### Tests
```sh
./build/src/tests/check_kernel
```
```sh
./build/src/tests/check_grad
```

### Plot
```sh
python (python3) scripts/plot.py rsc/rbf_spherical.csv
```
