# DomainStudio
DomainStudio is the proposed few-shot domain-driven generation method for diffusion models (preprint: arxiv 2306:14153). It is compatible with unconditional DDPMs and conditional text-to-image models (Stable Diffusion). The code are provided in the two files separately.

## Unconditional
Our code is based on [openai/improved-diffusion]. We train source models on FFHQ and LSUN-Church and adapt them to several target domians using 10-shot limited data.

## Text-to-Image
Our code is based on [huggingface/diffusers]. We use Stable Diffusion V1.4 as the source model and realize domain-driven generation using limited data.

