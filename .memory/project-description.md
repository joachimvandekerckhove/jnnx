# Calling neural networks inside JAGS

We are creating a program that can take a trained network (as a .onnx file and a
.pth file with scalers) and a control file (as a .json file), in order to create
a JAGS module.  The JAGS module should serve to add a single function to JAGS,
which simply evaluates the trained network as a deterministic node.  We will
assume that these three files are placed together in a folder, whose name ends
in `.jnnx`, and the three files have the same name stem.

The JSON control file should let us, at a minimum, set:

 - The number of input variables to the neural network
 - The number of output variables from the neural network
 - The limits of the input variables (some limits may be infinite)
 - The limits of the output variables (some limits may be infinite)
 - The name of the JAGS function
 - The name of the module
 - A banner string that is printed to screen when the module is loaded
 - The location of the JAGS modules directory to where the compiled module will
   be saved


## Phase 1: An interface to read and edit the JSON control file

In this Phase, we will create a Python script that can read the control file and
allows the user to edit any and all fields in it.  It should be called in a way
much like this:

    ./jnnx-setup models/sdt.jnnx/

And this should find the .json file in the .jnnx directory, print its current
values to the console, and let the user select a field to edit.  If a field is
very long, it should be clipped reasonably.


## Phase 2: Validate JNNX

In this Phase, we will create a test suite to validate a .jnnx folder.  The
test suite will be called like this:

    ./validate-jnnx models/sdt.jnnx/

Here are some conditions that should be met:

 1) The neural network should load from the .onnx and .pth files without error.
 2) Calling the neural network with a valid input vector (right size, within
    bounds) should work without error for a range of values.
 3) Calling the neural network with a valid input vector (right size, within
    bounds) should return a valid output vector (right size, within bounds).
 4) Calling the neural network with an invalid input vector (wrong size) should
    trigger an error.
 5) The installation directory should exist.
 
 The validation script should test all of these things and give concise but
 informative output to the console.


## Phase 3: Create the module code

In order to create modules that execute neural networks, we will need some
template code for a basic module.  Our program should copy and edit the template
with information from the JSON file, setting the filenames of the .onnx and .pth
files, setting the input and output dimensions, setting the name of the module,
hard-coding the string of text that should be printed on module load, and some
input checking.

We also need a Makefile that compiles and installs the module.  We will create
the Makefile also from a template.  The installation directory can be read from
the JSON file.


## Phase 4: Create an example module

We have example .jnnx folders in the models/ directory.  Let's use 
`models/sdt.jnnx/` as an example.  Currently the JSON file is empty, here is
the information that should go in it:

 - The number of input variables: 2
 - The number of output variables: 2
 - The limits of the input variables: [-Inf, -Inf] to [Inf, Inf]
 - The limits of the output variables: [0, 0] to [1, 1]
 - The name of the JAGS function: sdt
 - The name of the module: sdt_emulator
 - A banner string that is printed to screen when the module is loaded: 
   "The SDT emulator is being loaded. (c) 2025 Joachim Vandekerckhove"
 - The location of the JAGS modules directory to where the compiled module will
   be saved: /usr/lib/x86_64-linux-gnu/JAGS/modules-4/

We then want to create the example module.  First we would need to check that
the output file isn't already there -- if so, it needs to be cleaned away before
compiling.

Environment assumptions:
- JAGS == 4.3.0
- Python == 3.11.6 with torch, onnx, onnxruntime
- GCC >= 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04)

The compilation should of course complete with no errors.


## Phase 5: Create a validation suite for the module

In this Phase, we will create a test suite to validate a module.  The test suite
will be called like this:

    ./validate-module models/sdt.jnnx/

Here are some conditions that should be met:

 1) The neural network should load in JAGS without error.
    Specifically, when we call `echo "load $MODULENAME" | jags` from the command
    line, the output should contain the line `Loading module: $MODULENAME: ok`
    and the banner string should print.
 2) Calling the neural network with a valid input vector (right size, within
    bounds) from within JAGS should work without error for a range of values.
 3) Calling the neural network with a valid input vector (right size, within
    bounds) from within JAGS should return a valid output vector (right size, 
    within bounds).
 4) Calling the neural network with an invalid input vector (wrong size) from 
    within JAGS should trigger an error.
 5) Calling the neural network with an invalid input vector (out of bounds) from
    within JAGS should trigger an error.
 6) Calling the neural network with a valid input vector (right size, within
    bounds) from within JAGS should return an output vector that is numerically
    identical to the output obtained from the neural network if it is evaluated
    directly in Python.
 
The validation script should test all of these things and give concise but
informative output to the console.

