# Instant Neural Graphics Primitives ![](https://github.com/NVlabs/instant-ngp/workflows/CI/badge.svg)

## This is a forked Windows Installation Tutorial and the main codes will not be updated
Follow this YouTube [tutorial](https://youtu.be/kq9xlvz73Rg) to understand the installation process more easily and if you have any questions feel free to join my [discord](https://discord.gg/sE8R7e45MV) and ask there.

<img src="docs/assets_readme/fox.gif" height="342"/> <img src="docs/assets_readme/robot5.gif" height="342"/>

Ever wanted to train a NeRF model of a fox in under 5 seconds? Or fly around a scene captured from photos of a factory robot? Of course you have!

Here you will find an implementation of four __neural graphics primitives__, being neural radiance fields (NeRF), signed distance functions (SDFs), neural images, and neural volumes.
In each case, we train and render a MLP with multiresolution hash input encoding using the [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) framework.

> __Instant Neural Graphics Primitives with a Multiresolution Hash Encoding__  
> [Thomas Müller](https://tom94.net), [Alex Evans](https://research.nvidia.com/person/alex-evans), [Christoph Schied](https://research.nvidia.com/person/christoph-schied), [Alexander Keller](https://research.nvidia.com/person/alex-keller)  
> _[arXiv:2201.05989 [cs.CV]](https://arxiv.org/abs/2201.05989), Jan 2022_  
> __[&nbsp;[Project page](https://nvlabs.github.io/instant-ngp)&nbsp;] [&nbsp;[Paper](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf)&nbsp;] [&nbsp;[Video](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.mp4)&nbsp;] [&nbsp;[BibTeX](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.bib)&nbsp;]__

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)


## Requirements
- An __NVIDIA GPU__; tensor cores increase performance when available. All shown results come from an RTX 3090.
- Python ver: 3.9.*
- [Visual Studio Community 2019](https://docs.microsoft.com/en-us/visualstudio/releases/2019/history) (Latest the best, ~8GB) Below are the install requirements
![image](https://user-images.githubusercontent.com/29135514/151634222-6ac236c9-5fa7-4762-9144-73e50959cb65.png)
- __[CUDA](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_network) v11.6__. You can check ur CUDA version via `nvcc --version` in any prompt and if it's not CUDA11.6, refer to [this](https://github.com/bycloudai/SwapCudaVersionWindows) to swap/install the correct version. 
- On some machines, `pyexr` refuses to install via `pip`. This can be resolved by installing OpenEXR from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#openexr). See later.
- This installation tutorial will be using Anaconda. Download anaconda prompt [here](https://www.anaconda.com/products/individual).
- __[OptiX](https://developer.nvidia.com/optix) 7.3 or higher__ for faster mesh SDF training. You need to either login or join to obtain the installer. Set the system environment variables `OptiX_INSTALL_DIR` to the installation directory if it is not discovered automatically. Should look like this: ![image](https://user-images.githubusercontent.com/29135514/151631220-7a934f5c-c299-41ab-a44e-184e2dc142b9.png)

## Compilation

copy these files `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\extras\visual_studio_integration\MSBuildExtensions`
to here `C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Microsoft\VC\v160\BuildCustomizations`

`cd` into a directory that you want to download the codes at. Eg. `cd F:\Tutorial\ngp\` 

Begin by cloning this repository and all its submodules using the following command (if you don't have git, download [here](https://git-scm.com/download/win) and add to path):
```sh
$ git clone --recursive https://github.com/nvlabs/instant-ngp
$ cd instant-ngp
```
if your python is not 3.9 (check with command `python --version`) then you need to run the following command to get it to ver 3.9.*
```
conda install python=3.9
```

Then, open **Developer Command Prompt**, you can find this in your search bar.

![image](https://user-images.githubusercontent.com/29135514/151631759-ff8538ab-74c6-4c7b-962e-d7b097e819db.png)

Then `cd` to where you cloned your repository so you are in its root folder `/instant-ng/`:
```sh
cmake . -B build
cmake --build build --config RelWithDebInfo -j 16
```

If the any of these build fails, please consult [this list of possible fixes](https://github.com/NVlabs/instant-ngp#troubleshooting-compile-errors) before opening an issue.

If automatic GPU architecture detection fails, (as can happen if you have multiple GPUs installed), set the  `TCNN_CUDA_ARCHITECTURES` enivonment variable for the GPU you would like to use. The following table lists the values for common GPUs. If your GPU is not listed, consult [this exhaustive list](https://developer.nvidia.com/cuda-gpus).

| RTX 30X0 | A100 | RTX 20X0 | TITAN V / V100 | GTX 10X0 / TITAN Xp | GTX 9X0 | K80 |
|----------|------|----------|----------------|---------------------|---------|-----|
|       86 |   80 |       75 |             70 |                  61 |      52 |  37 |


## Interactive Training and Rendering on Custom Image Sets
Install [COLMAP](https://github.com/colmap/colmap/releases/tag/3.7), I used ver 3.7

Add it to your system environment variables at Environment Variables > System Variables Path > Edit environment variable

![image](https://user-images.githubusercontent.com/29135514/151633058-e45f9220-c417-4249-aff3-09d29c1a4e9b.png)

open anaconda prompt, if you don't have you don't have you can get it [here](https://www.anaconda.com/products/individual)
`cd` into isntant-ngp as root
```sh
conda create -n ngp python=3.9
conda activate ngp
pip install -r requirements.txt
```
if `pyexr` cannot be installed via `pip install pyexr`, download [OpenEXR‑1.3.2‑cp39‑cp39‑win_amd64.whl](https://www.lfd.uci.edu/~gohlke/pythonlibs/#openexr) and move it to your root folder. Then you can run:
```
pip install OpenEXR-1.3.2-cp39-cp39-win_amd64.whl
```
Place your custom image set under `data/<image_set_name>`

Get `transform.json` from the following command. Insert your path to your images at `<image/path>`
```sh
python scripts/colmap2nerf.py --colmap_matcher exhaustive --run_colmap --aabb_scale 16 --images <image/path>
```
`transform.json` will be generated at the root folder, drag and drop it into your `data/<image_set_name>` folder.

You have to reorganize the folder structure due to how `transforms.json` is created...

For example:

File Structure **BEFORE** generating transform.json
```
📂instant-ngp/ # this is root
├── 📂data/
│	├── 📂toy_truck/
│	│	├── 📜toy_truck_001.jpg
│	│	├── 📜toy_truck_002.jpg
│	│	│...
│   │...
│...
```

File Structure **AFTER** generating transform.json
```
📂instant-ngp/ # this is root
├── 📂data/
│	├── 📂toy_truck/
│	│	├── 📜transforms.json/
│	│	├── 📂data/
│	│	│	├── 📂toy_truck/
│	│	│	│	├── 📜toy_truck_001.jpg
│	│	│	│	├── 📜toy_truck_002.jpg
│	│	│	│	│...
│	│	│	│...
│	│	│...
│	│...
│...
```
Note: adjusting the `"aabb_scale"` inside `transform.json` can reduce load on GPU VRAM. The lower the value the less intensive it'll be. 


Finally, to run instant-ngp: 
```sh
<path_to_your_ngp>\instant-ngp\build\testbed.exe --scene data/<image_set_name>
```
eg.
```
C:\user\user\download\instant-ngp\build\testbed.exe --scene data/toy_truck
```

And it should launch the GUI and everything amazing with it

## Rendering custom camera path
1. May need to install more dependencies. Install `pip install tqdm scipy pillow opencv-python`, `conda install -c conda-forge ffmpeg`, might be needed in the conda virtual environment. Refer to installation of `pyexr` above in the installation section if you didn't install that too.
2. Train any image set like above. 
3. After you have reached a point that you are satisfied with your training, save a Snapshot on the GUI. (one of the tabs & no need to edit the path & the name)
4. Find another GUI called camera path, it'll play hide and seek with you but it is there so find that window.
5. The GUI is so well made, if you know how to use any 3D engine, it's really similar. Add camera path will give you a new angle of the camera. 
6. After you have finished adding your camera points, save the camera path. (no need to edit the path & the name)
7. Render the path with the following command:

```sh
python scripts/render.py --scene <scene_path> --n_seconds <seconds> --fps <fps> --render_name <name> --width <resolution_width> --height <resolution_height>

```

eg.
```sh
python scripts/render.py --scene data/toy --n_seconds 5 --fps 60 --render_name test --width 1920 --height 1080
```

Your video will be saved at root. You might have to play around with the `fps` and `n_seconds` to speed up or slow down. I couldn't get it accurately because of the lack of information and this is the best I could come up with. To be honest, this is only a short-term solution too, since the author has promised to publish an official one. So stay tuned!

And my fork edits end here.


## Interactive training and rendering

<img src="docs/assets_readme/testbed.png" width="100%"/>

This codebase comes with an interactive testbed that includes many features beyond our academic publication:
- Additional training features, such as extrinsics and intrinsics optimization.
- Marching cubes for `NeRF->Mesh` and `SDF->Mesh` conversion.
- A spline-based camera path editor to create videos.
- Debug visualizations of the activations of every neuron input and output.
- And many more task-specific settings.
- See also our [one minute demonstration video of the tool](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.mp4).



### NeRF fox

One test scene is provided in this repository, using a small number of frames from a casually captured phone video:

```sh
instant-ngp$ ./build/testbed --scene data/nerf/fox
```

<img src="docs/assets_readme/fox.png"/>

Alternatively, download any NeRF-compatible scene (e.g. [from the NeRF authors' drive](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi)).
Now you can run:

```sh
instant-ngp$ ./build/testbed --scene data/nerf_synthetic/lego/transforms_train.json
```

For more information about preparing datasets for use with our NeRF implementation, please see [this document](docs/nerf_dataset_tips.md).

### SDF armadillo

```sh
instant-ngp$ ./build/testbed --scene data/sdf/armadillo.obj
```

<img src="docs/assets_readme/armadillo.png"/>

### Image of Einstein

```sh
instant-ngp$ ./build/testbed --scene data/image/albert.exr
```

<img src="docs/assets_readme/albert.png"/>

To reproduce the gigapixel results, download, for example, [the Tokyo image](https://www.flickr.com/photos/trevor_dobson_inefekt69/29314390837) and convert it to `.bin` using the `scripts/image2bin.py` script. This custom format improves compatibility and loading speed when resolution is high. Now you can run:

```sh
instant-ngp$ ./build/testbed --scene data/image/tokyo.bin
```


### Volume Renderer

Download the [nanovdb volume for the Disney cloud](https://drive.google.com/drive/folders/1SuycSAOSG64k2KLV7oWgyNWyCvZAkafK?usp=sharing), which is derived [from here](https://disneyanimation.com/data-sets/?drawer=/resources/clouds/) ([CC BY-SA 3.0](https://media.disneyanimation.com/uploads/production/data_set_asset/6/asset/License_Cloud.pdf)).

```sh
instant-ngp$ ./build/testbed --mode volume --scene data/volume/wdas_cloud_quarter.nvdb
```
<img src="docs/assets_readme/cloud.png"/>


## Python bindings

To conduct controlled experiments in an automated fashion, all features from the interactive testbed (and more!) have Python bindings that can be easily instrumented.
For an example of how the `./build/testbed` application can be implemented and extended from within Python, see `./scripts/run.py`, which supports a superset of the command line arguments that `./build/testbed` does.

Happy hacking!


## Troubleshooting compile errors

Before investigating further, make sure all submodules are up-to-date and try compiling again.
```sh
instant-ngp$ git submodule sync --recursive
instant-ngp$ git submodule update --init --recursive
```
If __instant-ngp__ still fails to compile, update CUDA as well as your compiler to the latest versions you can install on your system. It is crucial that you update _both_, as newer CUDA versions are not always compatible with earlier compilers and vice versa.
If your problem persists, consult the following table of known issues.

| Problem | Resolution |
|---------|------------|
| __CMake error:__ No CUDA toolset found / CUDA_ARCHITECTURES is empty for target "cmTC_0c70f" | __Windows:__ the Visual Studio CUDA integration was not installed correctly. Follow [these instructions](https://github.com/mitsuba-renderer/mitsuba2/issues/103#issuecomment-618378963) to fix the problem without re-installing CUDA. ([#18](https://github.com/NVlabs/instant-ngp/issues/18)) |
| | __Linux:__ Environment variables for your CUDA installation are probably incorrectly set. You may work around the issue using ```cmake . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-<your cuda version>/bin/nvcc``` ([#28](https://github.com/NVlabs/instant-ngp/issues/28)) |
| __CMake error:__ No known features for CXX compiler "MSVC" | Reinstall Visual Studio & make sure you run CMake from a developer shell. ([#21](https://github.com/NVlabs/instant-ngp/issues/21)) |
| __Compile error:__ undefined references to "cudaGraphExecUpdate" / identifier "cublasSetWorkspace" is undefined | Update your CUDA installation (which is likely 11.0) to 11.3 or higher. ([#34](https://github.com/NVlabs/instant-ngp/issues/34) [#41](https://github.com/NVlabs/instant-ngp/issues/41) [#42](https://github.com/NVlabs/instant-ngp/issues/42)) |
| __Compile error:__ too few arguments in function call | Update submodules with the above two `git` commands. ([#37](https://github.com/NVlabs/instant-ngp/issues/37) [#52](https://github.com/NVlabs/instant-ngp/issues/52)) |
| __Python error:__ No module named 'pyngp' | It is likely that CMake did not detect your Python installation and therefore did not build `pyngp`. Check CMake logs to verify this. If `pyngp` was built in a different folder than `instant-ngp/build`, Python will be unable to detect it and you have to supply the full path to the import statement. ([#43](https://github.com/NVlabs/instant-ngp/issues/43)) |

If you cannot find your problem in the table, please feel free to [open an issue](https://github.com/NVlabs/instant-ngp/issues/new) and ask for help.

## Thanks

Many thanks to [Jonathan Tremblay](https://research.nvidia.com/person/jonathan-tremblay) and [Andrew Tao](https://developer.nvidia.com/blog/author/atao/) for testing early versions of this codebase and to Arman Toorians and Saurabh Jain for the factory robot dataset.
We also thank [Andrew Webb](https://github.com/grey-area) for noticing that one of the prime numbers in the spatial hash was not actually prime; this has been fixed since.

This project makes use of a number of awesome open source libraries, including:
* [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) for fast CUDA MLP networks
* [tinyexr](https://github.com/syoyo/tinyexr) for EXR format support
* [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader) for OBJ format support
* [stb_image](https://github.com/nothings/stb) for PNG and JPEG support
* [Dear ImGui](https://github.com/ocornut/imgui) an excellent immediate mode GUI library
* [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) a C++ template library for linear algebra
* [pybind11](https://github.com/pybind/pybind11) for seamless C++ / Python interop
* and others! See the `dependencies` folder.

Many thanks to the authors of these brilliant projects!

## License and Citation

```bibtex
@article{mueller2022instant,
    title = {Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    author = {Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
    journal = {arXiv:2201.05989},
    year = {2022},
    month = jan
}
```

Copyright © 2022, NVIDIA Corporation. All rights reserved.

This work is made available under the Nvidia Source Code License-NC. Click [here](LICENSE.txt) to view a copy of this license.
