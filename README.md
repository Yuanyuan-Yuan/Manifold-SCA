# Manifold-SCA
Research Artifact of USENIX Security 2022 Paper: *Automated Side Channel Analysis of Media Software with Manifold Learning*

The repo is organized as:

```
📂manifold-sca
 ┣ 📂vulnerability
 ┃ ┣ 📂contribution
 ┃ ┣ 📜{dataset}-{program}-count.json
 ┃ ┗ 📜{program}.dis
 ┣ 📂code
 ┃ ┣ 📂SCA
 ┃ ┣ 📂tools
 ┃ ┗ 📂pp
 ┣ 📂audio
 ┗ 📂output
```

## Code

We release our code in folder `code`. The implementation of our framework is in
folder `code/SCA` and tools we use to process input/output data are listed in
folder `code/tools`. To launch Prime+Prob, you can use the code in `code/pp`.

## Attack

To prepare the training data for learning data manifold, you first need to instrument
the *binary* with the released pintool `code/tools/pinatrace.cpp`. You will get
a sequence of *instruction address: accessed address* when the *binary* processes a
media data.
Then you need to fold the sequence of *accessed address* into a matrix and convert
the matrix with correct format (e.g., tensor, or numpy array).

We release the scripts for training the framework in folder `code/SCA`. Before training
you need to first customize data paths in each script. The training procedure ends
after 100 epochs and takes less than 24 hours on one Nvidia GeForce RTX 2080 GPU.

## Localize

Recall that we localize vulnerabilities by pinpointing records in a trace that contribute
most to reconstructing media data. So, to perform localization, you need first
train the framework as we introduced before.

After training the framework, you just need to run `code/localize.py` and `code/pinpoint.py`
to localize records in a side channel trace. Note that what you get in this step are several
*accessed addresses* with their indexes in the trace. You need further get the corresponding
*instruction addresses* based on the instrument output you generated when preparing
training data.

We release the localized vulnerabilities in folder `vulnerability`. In folder
`vulnerability/contribution`, we list the corresponding *instruction addresses* of records
that make primary contribution to the reconstruction of media data. We further map
the pinpoined instructions back to the corresponding functions. These functions are
regarded as **side-channel vulnerable functions**. We list the results in
`{dataset}-{program}-count.json`, where higher counting indicates a higher possibility
of being vulnerable.

Despite each program is evaluated on different datasets, we can still observe that highly
consistent vulnerabilities are localized in the same program.

## Prime+Probe

We use [Mastik](https://cs.adelaide.edu.au/~yval/Mastik/) to launch Prime+Probe on
L1 cache of Intel Xeon CPU and AMD Ryzen CPU. We release our scripts in folder `code/pp`.

The experiment is launched in Linux OS. You need first to install **taskset** and **cpuset**.

We assume *victim* and *spy* are on the same CPU core and no other process is runing
on this CPU core. To isolate a CPU core, you need to run *sudo cset shield --cpu {cpu_id}*.

Then run *sudo cset shield --exec python run_pp.py -- {cpu_id} {segment_id}*. Note that we
seperate the media data into several segments to speed up the side channel collection.
`code/pp/run_pp.py` runs `code/pp/pp_audio.py` with **taskset**.
`code/pp/pp_audio.py` is the coordinator which runs *spy* and *victim* on the same CPU
core simultaneously and saves the collected cache set access.

## Audio

We upload all (total 2,552) audios reconstructed by our framework under Prime+Probe to folder `audio/sc09-pp`
for result verification. Each audio is named as `{Number}_{hash}_{index}.wav` and the
`{Number}` is the content of the corresponding reference input, e.g., for
a reconstructed audio `One_94de6a6a_nohash_1.wav`, the number said in the reference input
is *one*. As we reported in the paper, most (~80%) of the audios have
consistent contents (i.e., the numbers) with the reference inputs.

## Output

We upload media data reconstructed by our framework in folder `output`.