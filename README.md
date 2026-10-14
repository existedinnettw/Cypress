# Cypress: VLSI-Inspired Scalable GPU-Accelerated PCB Placement

Cypress is a scalable, GPU-accelerated PCB placement method inspired by VLSI. It incorporates
tailored cost functions, constraint handling, and optimized techniques adapted for PCB layouts.

<p align="center">
  <img src="images/big-3.gif" width="60%" alt="Cypress PCB Placement Animation">
</p>


## How to Build

You can build in two ways:

- Build without Docker by following the instructions of the DREAMPlace build at README_DREAMPlace.md.
- Use the provided Dockerfile to build an image with the required library dependencies.

## Run PCB Placement

```sh
cd /Cypress
python dreamplace/Placer.py path/to/your/config.json
```

## Use with KiCad

Cypress can place the footprints of a KiCad board (KiCad 7 or newer) directly,
either from a button in the PCB editor or from the command line. The
`cypress_kicad` package turns the board into a placement problem: each
footprint's courtyard becomes a box, pads become pins, and the board outline
becomes the placement region. It runs the Cypress engine, removes any overlaps
that remain, and moves and rotates the footprints on the board.

### 1. Get the engine

The engine needs PyTorch and compiled C++/CUDA ops, so it runs in its own
process, separate from KiCad's Python. You can use either option:

- **Docker** (easiest): `docker build -f docker/Dockerfile -t cypress:cpu .`
  For an NVIDIA GPU, build the CUDA image instead:
  ```sh
  docker build -f docker/Dockerfile -t cypress:cuda \
    --build-arg BASE=nvidia/cuda:11.7.1-devel-ubuntu22.04 \
    --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu117 .
  ```
- **Local build**: follow README_DREAMPlace.md and install to `<repo>/install`
  (`cmake .. -DCMAKE_INSTALL_PREFIX=../install && make install`). Then point
  the plugin at a Python that has PyTorch.

Without a GPU it runs on the CPU, including part rotation.

### 2. KiCad plugin

```sh
git submodule update --init --recursive   # only needed for a local build
python3 kicad_plugin/install.py           # symlinks into KiCad's plugin folder
```

Restart the PCB editor, then click the **Cypress Placer** toolbar button or
go to *Tools → External Plugins → Cypress Placer*. In the dialog, choose the
engine (`docker` or `local`) and the placement options, then click **Place**.
If any footprints are selected, you can move only those and leave the rest
fixed. Locked footprints always stay fixed. The result can be undone with
*Edit → Undo*. Settings are saved to `~/.config/cypress-kicad/settings.json`.

### 3. Command line

Run the CLI with a Python that can import `pcbnew`, such as KiCad's own Python.
It also searches the usual install locations; otherwise set `CYPRESS_PCBNEW_PATH`.

```sh
python3 -m cypress_kicad check --runner docker
python3 -m cypress_kicad place board.kicad_pcb -o placed.kicad_pcb --runner docker
python3 -m cypress_kicad place board.kicad_pcb -o placed.kicad_pcb --refs U1 U2 C1 C2   # move only these
python3 -m cypress_kicad export board.kicad_pcb -d bookshelf/    # Bookshelf files only
```

See `python3 -m cypress_kicad place --help` for all options, including
iterations, target density, rotation, ratsnest-crossing penalty, part spacing,
edge clearance, ignored nets, and the legalization grid.

Notes:
- Each part's courtyard is used as its box. A part without a courtyard uses
  its body bounding box instead. Footprints without pads are ignored.
- Footprints with no connected nets (mounting holes, fiducials) stay where
  they are, as do locked footprints. Lock edge connectors and other parts
  whose position is set by the enclosure before you run the placer.
- Rule areas set to disallow footprints (or pads) are treated as keep-outs on
  their copper layers (by bounding box). Use them for areas under modules,
  cards or heatsinks. Rule areas inside a footprint (a module's antenna area)
  move with the part and keep other parts out.
- Isolation domains: a custom rule such as
  `A.NetClass == 'Logic*' && B.NetClass != 'Logic*'` with a clearance splits
  the parts into two domains plus bridges (isolators, optocouplers). If the
  smaller domain has its own copper pour covering part of the board (an
  isolated ground island), placement runs twice: the other domain is kept the
  rule's gap away from that pour, then the smaller domain and the bridges are
  placed inside it (bridges may straddle a 3 mm band). Without such a pour the
  domains are reported but not enforced. `--no-domains` turns this off.
- Ground nets (`GND`/`GROUND`, configurable) are left out of wirelength,
  because they are usually connected by planes.
- Parts stay on the side of the board they are on. A through-hole part keeps
  its whole courtyard free on the other side. Unplated holes (alignment pegs),
  pads with copper on the other outer layer, and courtyards drawn on the other
  side block the other side too.
- When the board has no outline, the current spread of parts plus a 10% margin
  is used as the placement region.

## Run Hyperparameter search

```sh
./tuner/run_tuner.sh 1 1 	test/tune/pcb-configspace.json 	/path/to/aux_file 	test/tuner/tuner-ppa.json "" 	100 4 0 0 10 	./tuner 	./results/result_path
```

## Acknowledgement

Cypress builds upon and extends the following open-source projects:

- [DREAMPlace](https://github.com/limbo018/DREAMPlace): A deep learning toolkit-enabled VLSI placement framework that provides the foundation for our GPU-accelerated placement engine.

- [AutoDMP](https://github.com/NVlabs/AutoDMP): An automated design methodology platform that inspired aspects of our hyperparameter tuning approach.

We thank the authors and contributors of these projects for their valuable work that made Cypress possible.

## Citation

If you find this work useful, please cite the following paper:

```bibtex
@inproceedings{zhang2025cypress,
  title={Cypress: VLSI-Inspired PCB Placement with GPU Acceleration},
  author={Zhang, Niansong and Agnesina, Anthony and Shbat, Noor and Leader, Yuval and Zhang, Zhiru and Ren, Haoxing},
  booktitle={Proceedings of the 2025 International Symposium on Physical Design},
  pages={xx--xx},
  year={2025}
}
```