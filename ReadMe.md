# Target-oriented Opinion Words Extraction with Target-fused Neural Sequence Labeling

This repository is under construction.

## Dependency:
* python3
* pytorch 0.4

## How to run (Take the 14res dataset as example)

1. (optional) prepare the word embeddings: (we have prepared for you in the code/data/ directory.)
    1. put the glove embeddings in the code/embedding directory.
    2. run the script:
    ```
    python build_vocab_embed.py --ds 14res
    ```
2. in the code/ directory:
    ```
    python main.py --ds 14res
    ```

# 环境安装
conda install pytorch=0.4.0 torchtext -c pytorch
Channels:
 - pytorch
 - defaults
Platform: linux-64
Collecting package metadata (repodata.json): done
Solving environment: - warning  libmamba Added empty dependency for problem type SOLVER_RULE_UPDATE
failed

LibMambaUnsatisfiableError: Encountered problems while solving:
  - nothing provides cudatoolkit >=11.1,<11.2 needed by pytorch-1.10.0-py3.6_cuda11.1_cudnn8.0.5_0

Could not solve for environment specs
The following packages are incompatible
├─ pin-1 is installable and it requires
│  └─ python 3.6.* , which can be installed;
├─ pytorch 0.4.0**  is installable with the potential options
│  ├─ pytorch [0.4.0|1.5.1] would require
│  │  └─ python >=3.5,<3.6.0a0 , which conflicts with any installable versions previously reported;
│  ├─ pytorch [0.4.0|1.2.0|1.3.1] would require
│  │  └─ python >=2.7,<2.8.0a0 , which conflicts with any installable versions previously reported;
│  ├─ pytorch 0.4.0, which can be installed;
│  └─ pytorch 0.4.0 would require
│     └─ cudatoolkit 8.0.* , which does not exist (perhaps a missing channel);
└─ torchtext is not installable because there are no viable options
   ├─ torchtext 0.10.0 would require
   │  └─ pytorch 1.9.0  but there are no viable options
   │     ├─ pytorch [1.10.0|1.10.1|...|1.9.1] would require
   │     │  └─ cudatoolkit >=11.1,<11.2 , which does not exist (perhaps a missing channel);
   │     ├─ pytorch [1.10.0|1.10.1|...|1.9.1] would require
   │     │  └─ python >=3.7,<3.8.0a0 , which conflicts with any installable versions previously reported;
   │     ├─ pytorch [1.10.0|1.10.1|...|2.2.0] would require
   │     │  └─ python >=3.8,<3.9.0a0 , which conflicts with any installable versions previously reported;
   │     ├─ pytorch [1.10.0|1.10.1|...|2.2.0] would require
   │     │  └─ python >=3.9,<3.10.0a0 , which conflicts with any installable versions previously reported;
   │     └─ pytorch 1.9.0 conflicts with any installable versions previously reported;
   ├─ torchtext [0.10.0|0.10.1|...|0.9.1] would require
   │  └─ python >=3.7,<3.8.0a0 , which conflicts with any installable versions previously reported;
   ├─ torchtext [0.10.0|0.10.1|...|0.9.1] would require
   │  └─ python >=3.8,<3.9.0a0 , which conflicts with any installable versions previously reported;
   ├─ torchtext [0.10.0|0.10.1|...|0.17.0] would require
   │  └─ python >=3.9,<3.10.0a0 , which conflicts with any installable versions previously reported;
   ├─ torchtext 0.10.1 would require
   │  └─ pytorch 1.9.1  but there are no viable options
   │     ├─ pytorch [1.10.0|1.10.1|...|1.9.1], which cannot be installed (as previously explained);
   │     ├─ pytorch [1.10.0|1.10.1|...|1.9.1], which cannot be installed (as previously explained);
   │     ├─ pytorch [1.10.0|1.10.1|...|2.2.0], which cannot be installed (as previously explained);
   │     ├─ pytorch [1.10.0|1.10.1|...|2.2.0], which cannot be installed (as previously explained);
   │     └─ pytorch 1.9.1 conflicts with any installable versions previously reported;
   ├─ torchtext 0.11.0 would require
   │  └─ pytorch 1.10.0  but there are no viable options
   │     ├─ pytorch 1.10.0 conflicts with any installable versions previously reported;
   │     ├─ pytorch [1.10.0|1.10.1|...|1.9.1], which cannot be installed (as previously explained);
   │     ├─ pytorch [1.10.0|1.10.1|...|1.9.1], which cannot be installed (as previously explained);
   │     ├─ pytorch [1.10.0|1.10.1|...|2.2.0], which cannot be installed (as previously explained);
   │     └─ pytorch [1.10.0|1.10.1|...|2.2.0], which cannot be installed (as previously explained);
   ├─ torchtext 0.11.1 would require
   │  └─ pytorch 1.10.1  but there are no viable options
   │     ├─ pytorch [1.10.0|1.10.1|...|1.9.1], which cannot be installed (as previously explained);
   │     ├─ pytorch [1.10.0|1.10.1|...|1.9.1], which cannot be installed (as previously explained);
   │     ├─ pytorch [1.10.0|1.10.1|...|2.2.0], which cannot be installed (as previously explained);
   │     ├─ pytorch [1.10.0|1.10.1|...|2.2.0], which cannot be installed (as previously explained);
   │     └─ pytorch 1.10.1 conflicts with any installable versions previously reported;
   ├─ torchtext 0.11.2 would require
   │  └─ pytorch 1.10.2  but there are no viable options
   │     ├─ pytorch [1.10.0|1.10.1|...|1.9.1], which cannot be installed (as previously explained);
   │     ├─ pytorch [1.10.0|1.10.1|...|1.9.1], which cannot be installed (as previously explained);
   │     ├─ pytorch [1.10.0|1.10.1|...|2.2.0], which cannot be installed (as previously explained);
   │     ├─ pytorch [1.10.0|1.10.1|...|2.2.0], which cannot be installed (as previously explained);
   │     ├─ pytorch 1.10.2 conflicts with any installable versions previously reported;
   │     └─ pytorch [1.10.2|1.11.0|...|2.2.0] would require
   │        └─ python >=3.10,<3.11.0a0 , which conflicts with any installable versions previously reported;
   ├─ torchtext [0.12.0|0.13.0|...|0.17.0] would require
   │  └─ python >=3.10,<3.11.0a0 , which conflicts with any installable versions previously reported;
   ├─ torchtext [0.15.2|0.16.0|0.16.1|0.16.2|0.17.0] would require
   │  └─ python >=3.11,<3.12.0a0 , which conflicts with any installable versions previously reported;
   ├─ torchtext 0.7.0 would require
   │  └─ pytorch 1.6.0  but there are no viable options
   │     ├─ pytorch [1.10.0|1.10.1|...|1.9.1], which cannot be installed (as previously explained);
   │     ├─ pytorch [1.10.0|1.10.1|...|2.2.0], which cannot be installed (as previously explained);
   │     └─ pytorch 1.6.0 conflicts with any installable versions previously reported;
   ├─ torchtext 0.8.0 would require
   │  └─ pytorch 1.7.0  but there are no viable options
   │     ├─ pytorch [1.10.0|1.10.1|...|1.9.1], which cannot be installed (as previously explained);
   │     ├─ pytorch [1.10.0|1.10.1|...|2.2.0], which cannot be installed (as previously explained);
   │     └─ pytorch 1.7.0 conflicts with any installable versions previously reported;
   ├─ torchtext 0.8.1 would require
   │  └─ pytorch 1.7.1  but there are no viable options
   │     ├─ pytorch [1.10.0|1.10.1|...|1.9.1], which cannot be installed (as previously explained);
   │     ├─ pytorch [1.10.0|1.10.1|...|2.2.0], which cannot be installed (as previously explained);
   │     ├─ pytorch [1.10.0|1.10.1|...|2.2.0], which cannot be installed (as previously explained);
   │     ├─ pytorch 1.7.1 conflicts with any installable versions previously reported;
   │     └─ pytorch [1.7.1|1.8.0] would require
   │        └─ python_abi 3.9.* *_cp39, which does not exist (perhaps a missing channel);
   ├─ torchtext [0.8.1|0.9.0|0.9.1] would require
   │  └─ python_abi 3.9.* *_cp39, which does not exist (perhaps a missing channel);
   ├─ torchtext 0.9.0 would require
   │  └─ pytorch 1.8.0  but there are no viable options
   │     ├─ pytorch [1.10.0|1.10.1|...|1.9.1], which cannot be installed (as previously explained);
   │     ├─ pytorch [1.10.0|1.10.1|...|1.9.1], which cannot be installed (as previously explained);
   │     ├─ pytorch [1.10.0|1.10.1|...|2.2.0], which cannot be installed (as previously explained);
   │     ├─ pytorch [1.7.1|1.8.0], which cannot be installed (as previously explained);
   │     └─ pytorch 1.8.0 conflicts with any installable versions previously reported;
   ├─ torchtext 0.9.1 would require
   │  └─ pytorch 1.8.1  but there are no viable options
   │     ├─ pytorch [1.10.0|1.10.1|...|1.9.1], which cannot be installed (as previously explained);
   │     ├─ pytorch [1.10.0|1.10.1|...|1.9.1], which cannot be installed (as previously explained);
   │     ├─ pytorch [1.10.0|1.10.1|...|2.2.0], which cannot be installed (as previously explained);
   │     ├─ pytorch [1.10.0|1.10.1|...|2.2.0], which cannot be installed (as previously explained);
   │     └─ pytorch 1.8.1 conflicts with any installable versions previously reported;
   └─ torchtext [0.4.0|0.5.0|0.6.0] would require
      └─ pytorch >=1.2  but there are no viable options
         ├─ pytorch 1.10.0 conflicts with any installable versions previously reported;
         ├─ pytorch [1.10.0|1.10.1|...|1.9.1], which cannot be installed (as previously explained);
         ├─ pytorch [1.10.0|1.10.1|...|1.9.1], which cannot be installed (as previously explained);
         ├─ pytorch [1.10.0|1.10.1|...|2.2.0], which cannot be installed (as previously explained);
         ├─ pytorch [1.10.0|1.10.1|...|2.2.0], which cannot be installed (as previously explained);
         ├─ pytorch 1.10.1 conflicts with any installable versions previously reported;
         ├─ pytorch 1.10.2 conflicts with any installable versions previously reported;
         ├─ pytorch [1.10.2|1.11.0|...|2.2.0], which cannot be installed (as previously explained);
         ├─ pytorch 1.11.0 would require
         │  └─ cudatoolkit >=11.5,<11.6 , which does not exist (perhaps a missing channel);
         ├─ pytorch [1.12.0|1.12.1] would require
         │  └─ cudatoolkit >=11.6,<11.7 , which does not exist (perhaps a missing channel);
         ├─ pytorch [1.13.0|1.13.1] would require
         │  └─ pytorch-cuda >=11.6,<11.7 , which requires
         │     └─ cuda 11.6.* , which does not exist (perhaps a missing channel);
         ├─ pytorch [1.13.0|1.13.1|2.0.0|2.0.1] would require
         │  └─ pytorch-cuda >=11.7,<11.8  but there are no viable options
         │     ├─ pytorch-cuda 11.7 would require
         │     │  └─ cuda-cudart >=11.7,<11.8 , which does not exist (perhaps a missing channel);
         │     └─ pytorch-cuda 11.7 would require
         │        └─ cuda 11.7.* , which does not exist (perhaps a missing channel);
         ├─ pytorch [0.4.0|1.5.1], which cannot be installed (as previously explained);
         ├─ pytorch [1.2.0|1.3.1|1.4.0|1.5.1] conflicts with any installable versions previously reported;
         ├─ pytorch 1.6.0 conflicts with any installable versions previously reported;
         ├─ pytorch 1.7.0 conflicts with any installable versions previously reported;
         ├─ pytorch 1.7.1 conflicts with any installable versions previously reported;
         ├─ pytorch [1.7.1|1.8.0], which cannot be installed (as previously explained);
         ├─ pytorch 1.8.0 conflicts with any installable versions previously reported;
         ├─ pytorch 1.8.1 conflicts with any installable versions previously reported;
         ├─ pytorch 1.9.0 conflicts with any installable versions previously reported;
         ├─ pytorch 1.9.1 conflicts with any installable versions previously reported;
         ├─ pytorch [2.0.0|2.0.1|...|2.2.0] would require
         │  └─ pytorch-cuda >=11.8,<11.9  but there are no viable options
         │     ├─ pytorch-cuda 11.8 would require
         │     │  └─ cuda-cudart >=11.8,<12.0 , which does not exist (perhaps a missing channel);
         │     └─ pytorch-cuda 11.8 would require
         │        └─ cuda 11.8.* , which does not exist (perhaps a missing channel);
         ├─ pytorch [1.13.1|2.0.1|...|2.2.0] would require
         │  └─ python >=3.11,<3.12.0a0 , which conflicts with any installable versions previously reported;
         ├─ pytorch [2.1.0|2.1.1|2.1.2|2.2.0] would require
         │  └─ pytorch-cuda >=12.1,<12.2 , which requires
         │     └─ cuda-cudart >=12.1,<12.2 , which does not exist (perhaps a missing channel);
         ├─ pytorch 2.2.0 would require
         │  └─ python >=3.12,<3.13.0a0 , which conflicts with any installable versions previously reported;
         ├─ pytorch [0.4.0|1.2.0|1.3.1], which cannot be installed (as previously explained);
         └─ pytorch 2.0.1 would require
            └─ __cuda >=11.8 , which is missing on the system.

Pins seem to be involved in the conflict. Currently pinned specs:
 - python 3.6.* (labeled as 'pin-1')
