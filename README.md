# Transfer Learning Capable Symbolic Regression (TLC-SR)

This is the readme file for Transfer Learning Capable Symbolic Regression (TLC-SR).  TLC-SR trains a recurrent neural network to perform symbolic regression on multiple datasets.

## Setup
Create the environmental `$TLCSR_DATA`. Data generated from the algorithm will be stored at this location.

## My Settings
I can run this on mac using python 3.7.4. These scripts require the following packages: pycma, numpy, pandas, networkx, and keras (with tensorflow backend). I am currently using the following versions of these packages:
|   Package|Version|
|---------:|:-----:|
|     pycma| 2.7.0 |
|     numpy|1.17.1 |
|    pandas|0.25.1 |
|  networkx|  2.3  |
|     Keras| 2.3.1 |
|tensorflow|1.15.0 |

## Running the Algorithm
Data generated during training will be stored in `$TLCSR_DATA/experiment<exp>/` where `<exp>` is the experiment number which you will specify when running the algorithm. To train TLC-SR network run
```bash
python3 run_tlcsr.py <rep> <exp> --use_benchmarks --test_index <index> --use_kexpressions --simultaneous_targets
```

where `<rep>` and `<exp>` are positive integers corresponding to the experiment number and the repetition number inside that experiment and `<index>` is the index to the list of target functions, which chooses the test function. These numbers will be included in output filenames and/or locations.

Similarly, to run the control experiment -- genetic programming (GP) with age-fitness pareto optimization (AFPO) -- use the following command
```bash
python3 run_tlcsr.py <rep> <exp> --use_benchmarks --test_index <index> --use_kexpressions --simultaneous_targets --genetic_programming
```
To recreate my results exactly use `<exp> = 25` and repeat for `<rep> = 0` through `<rep> = 29` and repeat again for `<index> = 0` through `<index> = 5`.